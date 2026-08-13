# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consecutive n-gram blocking, for models that loop on degenerate audio.

ASR models sometimes fall into a repetition loop, emitting the same word or
phrase until the token budget runs out. This processor watches the tail of
each request's output and bans the token that would start yet another copy of
an n-gram that has already repeated too often.

Opt in per request through ``SamplingParams.extra_args`` (``vllm_xargs`` over
the REST API), and load it at startup with::

    --logits-processors \
        vllm.v1.sample.logits_processor.consecutive_ngram:ConsecutiveNgramBlockingLogitsProcessor
"""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig

PARAM_NAME = "consecutive_ngram_block_reps"
"""Per-request argument: max consecutive repeats allowed per n-gram size.

A list whose i-th entry caps repeats of ``(i + 1)``-grams; ``0`` disables that
size. ``[8, 8, 3, 3, 3]`` matches CrisperWhisper's defaults.
"""

MAX_NGRAM_SIZE = 16

# Per-request state: (repeat caps, live output token ids).
_ReqState = tuple[tuple[int, ...], Sequence[int]]


def _parse_reps(sampling_params: SamplingParams) -> tuple[int, ...] | None:
    extra_args = sampling_params.extra_args
    if not extra_args or (raw := extra_args.get(PARAM_NAME)) is None:
        return None

    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise VLLMValidationError(
            f"{PARAM_NAME} must be a list of integers",
            parameter=PARAM_NAME,
            value=raw,
        )
    if not 0 < len(raw) <= MAX_NGRAM_SIZE:
        raise VLLMValidationError(
            f"{PARAM_NAME} must hold between 1 and {MAX_NGRAM_SIZE} entries",
            parameter=PARAM_NAME,
            value=raw,
        )

    reps: list[int] = []
    for entry in raw:
        if isinstance(entry, bool) or not isinstance(entry, int):
            if not (isinstance(entry, float) and entry.is_integer()):
                raise VLLMValidationError(
                    f"{PARAM_NAME} entries must be integers",
                    parameter=PARAM_NAME,
                    value=entry,
                )
            entry = int(entry)
        if entry < 0:
            raise VLLMValidationError(
                f"{PARAM_NAME} entries must be non-negative",
                parameter=PARAM_NAME,
                value=entry,
            )
        reps.append(entry)

    return tuple(reps) if any(reps) else None


def compute_banned_tokens(
    output_token_ids: Sequence[int],
    reps: tuple[int, ...],
) -> set[int]:
    """Tokens that would start one repeat too many, given the current tail."""
    banned = set[int]()
    for index, max_reps in enumerate(reps):
        if max_reps <= 0:
            continue
        ngram_size = index + 1
        span = ngram_size * max_reps
        if len(output_token_ids) < span:
            continue
        tail = output_token_ids[-span:]
        gram = tuple(tail[-ngram_size:])
        if all(
            tuple(tail[i * ngram_size : (i + 1) * ngram_size]) == gram
            for i in range(max_reps)
        ):
            banned.add(gram[0])
    return banned


class ConsecutiveNgramBlockingLogitsProcessor(LogitsProcessor):
    """Ban tokens that would extend an already-repeating n-gram."""

    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        self.device = device
        self.req_state = dict[int, _ReqState]()

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        _parse_reps(sampling_params)

    def is_argmax_invariant(self) -> bool:
        """Banning the greedy token is the whole point."""
        return False

    @staticmethod
    def _new_state(
        params: SamplingParams,
        _prompt_token_ids: list[int] | None,
        output_token_ids: list[int],
    ) -> _ReqState | None:
        reps = _parse_reps(params)
        return None if reps is None else (reps, output_token_ids)

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.req_state, batch_update, self._new_state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for index, (reps, output_token_ids) in self.req_state.items():
            if banned := compute_banned_tokens(output_token_ids, reps):
                logits[index, list(banned)] = -math.inf
        return logits
