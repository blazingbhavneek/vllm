# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CrisperWhisper 2.0 (nyralabs/CrisperWhisper2.0_*).

Architecturally identical to Whisper; differs only in the decoder prompt, which
carries mode-control tags before ``<|startoftranscript|>``, and in one unused
training-time head in the checkpoint.
"""

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from vllm.config import ModelConfig, SpeechToTextConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, PromptType, TextPrompt
from vllm.multimodal import MULTIMODAL_REGISTRY

from .interfaces import StreamingTranscriptionPostProcessor
from .utils import AutoWeightsLoader
from .whisper import (
    WhisperDummyInputsBuilder,
    WhisperForConditionalGeneration,
    WhisperMultiModalProcessor,
    WhisperProcessingInfo,
    _create_fake_bias_for_k_proj,
)

MODE_TAG_COUNT = 5
VALID_MODES = ("verbatim", "intended")

# Coverage-gated temperature fallback, ported from the reference
# implementation's `crisperwhisper/fallback.py`. Greedy decoding sometimes
# collapses on a chunk, returning a confident near-empty transcript ("
# Meanwhile.") for 30s of dense speech; average-logprob gates miss it because
# the short output scores well. Speech coverage is the reliable signal.
FALLBACK_TEMPERATURES = (0.4, 0.6, 0.8, 1.0)
MIN_SPEECH_S = 5.0
MIN_WORDS_PER_S = 0.5
_ENERGY_FRAME_S = 0.01

# Longform geometry, from the reference `LongformConfig` defaults in
# crisperwhisper/longform/base.py. The 30s window is the encoder length
# (SpeechToTextConfig.max_audio_clip_s), so a 26s stride leaves a 4s overlap,
# and LONGFORM_CONTEXT_WORDS is sized so the context text spans that overlap.
LONGFORM_STRIDE_S = 26.0
LONGFORM_CONTEXT_WORDS = 12
LONGFORM_CONTEXT_CHARS = 40
# The reference default (timestamp_aware_drop=True) drops a trailing word only
# when its audio starts inside the overlap. That needs per-word timings from
# cross-attention DTW, which vLLM cannot expose, so this is the reference's
# timestamp_aware_drop=False path: a fixed drop at every non-final boundary.
LONGFORM_DROP_WORDS = 2

_PROMPT_ARTIFACT_RE = re.compile(
    r"\[verbatim_\d+\]|\[intended_\d+\]"
    r"|<ctx>.*?<ectx>|<htx>.*?<ehtx>|<vtx>.*?<evtx>",
    re.DOTALL,
)

_ARTIFACT_OPENERS = ("<ctx>", "<htx>", "<vtx>")


def _speech_active_seconds(audio: np.ndarray, sample_rate: float) -> float:
    """Seconds of speech-active audio, by percentile energy gate.

    The reference reads this off the log-mel; the serving layer only has the
    waveform, so the same gate is applied to short-time frame energy.
    """
    frame = max(1, int(sample_rate * _ENERGY_FRAME_S))
    num_frames = audio.shape[-1] // frame
    if num_frames == 0:
        return 0.0

    frames = audio[: num_frames * frame].astype(np.float32).reshape(num_frames, frame)
    energy = np.log10(np.mean(np.square(frames), axis=1) + 1e-10)
    floor, peak = np.percentile(energy, 10), np.percentile(energy, 95)
    if peak - floor < 1e-3:
        return 0.0
    return float(np.count_nonzero(energy > floor + 0.2 * (peak - floor))) * (
        _ENERGY_FRAME_S
    )


class CrisperWhisperStreamingPostProcessor(StreamingTranscriptionPostProcessor):
    """Strip prompt artifacts from streaming deltas.

    A marker can straddle a delta boundary, so text is held back until it is
    known not to start one. The concatenated deltas match what
    ``CrisperWhisperForConditionalGeneration.post_process_output`` returns for
    the same text.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._pending_space = False
        self._started = False

    def process_delta(self, text_delta: str, finished: bool) -> str:
        self._buffer = _PROMPT_ARTIFACT_RE.sub("", self._buffer + text_delta)

        if finished:
            ready, self._buffer = self._buffer, ""
        else:
            hold_from = self._hold_index(self._buffer)
            ready, self._buffer = self._buffer[:hold_from], self._buffer[hold_from:]

        return self._normalize_spacing(ready)

    @staticmethod
    def _hold_index(text: str) -> int:
        """First index whose remainder could still grow into an artifact."""
        # Complete artifacts are already gone, so a surviving opener is one
        # whose closing tag has not arrived yet.
        for opener in _ARTIFACT_OPENERS:
            if (idx := text.find(opener)) != -1:
                return idx

        # An unclosed trailing '<' or '[' may be a marker split across deltas.
        # '[' also opens vocal-event tokens such as '[UM]', which are content
        # and are only briefly delayed.
        for idx in range(len(text) - 1, -1, -1):
            if (char := text[idx]) in "<[":
                closer = ">" if char == "<" else "]"
                return idx if closer not in text[idx:] else len(text)
        return len(text)

    def _normalize_spacing(self, text: str) -> str:
        """Collapse whitespace runs without emitting a leading/trailing space."""
        if not (parts := text.split()):
            self._pending_space = self._pending_space or (self._started and bool(text))
            return ""

        out = " ".join(parts)
        if self._pending_space or (self._started and text[:1].isspace()):
            out = " " + out
        self._pending_space = text[-1:].isspace()
        self._started = True
        return out


class CrisperWhisperMultiModalProcessor(WhisperMultiModalProcessor):
    # The mode tags precede <|startoftranscript|>, which the prompts below
    # already carry; prepending another decoder start token would put the
    # sequence out of distribution.
    skip_decoder_start_token: bool = True


@MULTIMODAL_REGISTRY.register_processor(
    CrisperWhisperMultiModalProcessor,
    info=WhisperProcessingInfo,
    dummy_inputs=WhisperDummyInputsBuilder,
)
class CrisperWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    # Word timings come from cross-attention DTW, which vLLM cannot expose, and
    # the model is always prompted with <|notimestamps|>; timestamp-token
    # decoding is therefore unsupported.
    supports_segment_timestamp = False

    # The continuation objective is trained on *overlapping* windows: <ctx>
    # holds the words the next window is about to re-hear, and the model emits
    # only what follows them. Decoding disjoint windows instead makes <ctx>
    # describe audio the model never hears, which is out of distribution and
    # costs words at every seam. Reference: LongformConfig in
    # crisperwhisper/longform/base.py.
    supports_sequential_chunking = True
    sequential_stride_s = LONGFORM_STRIDE_S

    @classmethod
    def _mode_tags(cls, mode: str) -> str:
        return "".join(f"[{mode}_{i}]" for i in range(1, MODE_TAG_COUNT + 1))

    @classmethod
    def _resolve_mode(cls, stt_params: SpeechToTextParams) -> str:
        extra = stt_params.extra_args or {}
        mode = str(extra.get("crisper_mode", "verbatim"))
        if mode not in VALID_MODES:
            raise ValueError(
                f"Unsupported crisper_mode {mode!r}; expected one of {VALID_MODES}."
            )
        return mode

    @classmethod
    def get_generation_prompt(cls, stt_params: SpeechToTextParams) -> PromptType:
        if stt_params.task_type != "transcribe":
            raise ValueError(
                "CrisperWhisper 2.0 only supports the 'transcribe' task; it is "
                "not trained for translation."
            )
        language = stt_params.language
        if language is None:
            raise ValueError(
                "Language must be specified when creating the CrisperWhisper prompt"
            )

        # Tag order must match training: mode tags, then context, then hotwords,
        # then the standard Whisper prefix.
        decoder_text = cls._mode_tags(cls._resolve_mode(stt_params))
        context = " ".join(
            part.strip()
            for part in (stt_params.request_prompt, stt_params.previous_text)
            if part and part.strip()
        )
        if context:
            decoder_text += f" <ctx> {context} <ectx>"
        if hotwords := (stt_params.hotwords or "").strip():
            decoder_text += f" <htx> {hotwords} <ehtx>"
        decoder_text += (
            f"<|startoftranscript|><|{language}|><|transcribe|><|notimestamps|>"
        )

        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",
                multi_modal_data={
                    "audio": (stt_params.audio, stt_params.stt_config.sample_rate)
                },
            ),
            decoder_prompt=TextPrompt(prompt=decoder_text),
        )

    @classmethod
    def get_language_detection_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
    ) -> PromptType:
        # Keep the mode tags so the detection step stays in distribution.
        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",
                multi_modal_data={"audio": (audio, stt_config.sample_rate)},
            ),
            decoder_prompt=TextPrompt(
                prompt=cls._mode_tags("verbatim") + "<|startoftranscript|>"
            ),
        )

    @classmethod
    def post_process_output(cls, text: str) -> str:
        # Vocal-event tokens ([UM], [laughter], ...) are transcript content and
        # are deliberately not stripped.
        return " ".join(_PROMPT_ARTIFACT_RE.sub("", text).split())

    @classmethod
    def get_streaming_post_processor_cls(
        cls,
    ) -> type[StreamingTranscriptionPostProcessor]:
        return CrisperWhisperStreamingPostProcessor

    @classmethod
    def should_retry_chunk(
        cls,
        text: str,
        stt_params: SpeechToTextParams,
        attempt: int,
    ) -> dict[str, Any] | None:
        if attempt >= len(FALLBACK_TEMPERATURES):
            return None
        # Word counting is meaningless for languages written without spaces.
        if stt_params.language in cls.no_space_languages:
            return None

        speech_s = _speech_active_seconds(
            stt_params.audio, stt_params.stt_config.sample_rate
        )
        if speech_s < MIN_SPEECH_S or len(text.split()) >= MIN_WORDS_PER_S * speech_s:
            return None

        # Recovery is stochastic, so escalate rather than repeating a draw.
        return {"temperature": FALLBACK_TEMPERATURES[attempt]}

    @classmethod
    def get_chunk_context(
        cls,
        previous_text: str,
        stt_params: SpeechToTextParams,
    ) -> str | None:
        if not (previous_text := previous_text.strip()):
            return None
        if stt_params.language in cls.no_space_languages:
            return previous_text[-LONGFORM_CONTEXT_CHARS:]
        return " ".join(previous_text.split()[-LONGFORM_CONTEXT_WORDS:])

    @classmethod
    def trim_chunk_boundary(cls, text: str, stt_params: SpeechToTextParams) -> str:
        # A window boundary can cut a word in half; the next window re-hears
        # that audio, so the tail is handed to it rather than kept mangled.
        if stt_params.language in cls.no_space_languages:
            return text
        words = text.split()
        if len(words) <= LONGFORM_DROP_WORDS:
            return text
        return " ".join(words[:-LONGFORM_DROP_WORDS])

    @classmethod
    def get_suppress_token_ids(cls, model_config: ModelConfig) -> list[int] | None:
        # The checkpoint ships a suppress list covering the prompt-control
        # markup; HF applies it on every step. `--generation-config vllm` asks
        # for neutral defaults, so honor that as the opt-out.
        if model_config.generation_config == "vllm":
            return None

        suppress_tokens = model_config.try_get_generation_config().get(
            "suppress_tokens"
        )
        if not suppress_tokens:
            return None

        vocab_size = model_config.get_vocab_size()
        return [int(t) for t in suppress_tokens if 0 <= int(t) < vocab_size]

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        config = super().get_speech_to_text_config(model_config, task_type)
        # Sequential decoding is how this model is meant to be run; the
        # concurrent path stays reachable via transcribe_mode='fast'.
        config.sequential_chunking = True
        return config

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # encoder_blank_head is a training-time alignment head with no
        # inference-time module.
        loader = AutoWeightsLoader(
            self, skip_prefixes=["proj_out.", "encoder_blank_head."]
        )
        weights = _create_fake_bias_for_k_proj(weights, ".k_proj.weight")
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
