# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Word timings travelling from the workers into the response.

The export deliberately avoids the engine's output structures, riding
`collective_rpc` instead, so the pieces below are the whole contract: a model
that says how to ask, and a merge that puts each chunk's window-local seconds
back on the audio timeline.
"""

import pytest

from vllm.entrypoints.speech_to_text.base.serving import _merge_chunk_word_timings
from vllm.model_executor.models.crisperwhisper import (
    WORD_TIMING_CAPABILITY,
    CrisperWhisperForConditionalGeneration,
    capture_key,
)
from vllm.model_executor.models.interfaces import SupportsTranscription
from vllm.model_executor.models.whisper import WhisperForConditionalGeneration


def test_timings_are_absent_by_default():
    assert _merge_chunk_word_timings(None, [0.0]) is None
    assert _merge_chunk_word_timings({}, [0.0]) is None
    # A chunk that produced no words must not invent an empty list.
    assert _merge_chunk_word_timings({0: []}, [0.0]) is None


def test_each_chunk_is_shifted_onto_the_audio_timeline():
    """Chunk 1 starts one stride in, so its seconds must move by that much."""
    merged = _merge_chunk_word_timings(
        {
            0: [{"word": "mary", "start": 0.10, "end": 0.40}],
            1: [{"word": "lamb", "start": 0.20, "end": 0.50}],
        },
        [0.0, 26.0],
    )

    assert [w.word for w in merged] == ["mary", "lamb"]
    assert merged[0].start == pytest.approx(0.10)
    assert merged[1].start == pytest.approx(26.20)
    assert merged[1].end == pytest.approx(26.50)


def test_chunks_are_emitted_in_audio_order():
    merged = _merge_chunk_word_timings(
        {2: [{"word": "c", "start": 0.0, "end": 0.1}],
         0: [{"word": "a", "start": 0.0, "end": 0.1}],
         1: [{"word": "b", "start": 0.0, "end": 0.1}]},
        [0.0, 26.0, 52.0],
    )

    assert [w.word for w in merged] == ["a", "b", "c"]
    assert [round(w.start, 1) for w in merged] == [0.0, 26.0, 52.0]


def test_missing_offset_does_not_raise():
    merged = _merge_chunk_word_timings(
        {5: [{"word": "x", "start": 1.0, "end": 2.0}]}, [0.0]
    )
    assert merged[0].start == pytest.approx(1.0)


def test_only_models_that_opt_in_are_timed():
    assert CrisperWhisperForConditionalGeneration.supports_word_timings is True
    assert WhisperForConditionalGeneration.supports_word_timings is False
    assert SupportsTranscription.supports_word_timings is False


def test_models_that_opt_out_ask_for_nothing():
    """The serving layer skips timing when the hook returns None."""
    assert (
        SupportsTranscription.build_word_timing_rpc([1, 2], ["a", "b"]) is None
    )
    assert SupportsTranscription.get_alignment_prompt(None, "text") is None


def test_rpc_plan_names_the_pass_and_its_words():
    ids = [50258, 100, 101, 50257]
    pieces = ["<|startoftranscript|>", "Mary", " had", "<|endoftext|>"]

    plan = CrisperWhisperForConditionalGeneration.build_word_timing_rpc(ids, pieces)

    capability, args, words = plan
    # Only a name crosses the engine boundary; a callable would need
    # VLLM_ALLOW_INSECURE_SERIALIZATION, and the workers resolve this on
    # the model via `WorkerBase.call_model_capability`.
    assert capability == WORD_TIMING_CAPABILITY
    assert isinstance(capability, str)
    assert callable(
        getattr(CrisperWhisperForConditionalGeneration, capability, None)
    )
    assert words == ["Mary", "had"]
    key_id, groups, token_ids = args
    assert key_id == capture_key(ids)
    assert token_ids == ids
    # Specials are excluded; each word keeps its own token positions.
    assert groups == [[1], [2]]


def test_rpc_plan_crosses_the_engine_boundary():
    """The plan must encode without insecure serialization being enabled.

    A callable here encodes only when VLLM_ALLOW_INSECURE_SERIALIZATION=1, so
    the whole feature would silently produce no words on a default server.
    """
    from vllm.v1.serial_utils import MsgpackEncoder

    plan = CrisperWhisperForConditionalGeneration.build_word_timing_rpc(
        [50258, 100, 101, 50257],
        ["<|startoftranscript|>", "Mary", " had", "<|endoftext|>"],
    )
    capability, args, _ = plan

    # Exactly what `collective_rpc` puts on the wire.
    MsgpackEncoder().encode((0, 1, "call_model_capability", (capability, args)))


def test_workers_expose_the_capability_hook():
    """The string form resolves on the worker, whatever the executor."""
    from vllm.v1.worker.worker_base import WorkerBase

    assert callable(WorkerBase.call_model_capability)


def test_rpc_plan_is_none_without_words():
    plan = CrisperWhisperForConditionalGeneration.build_word_timing_rpc(
        [50258], ["<|startoftranscript|>"]
    )
    assert plan is None
