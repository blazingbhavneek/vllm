# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CrisperWhisper 2.0 (nyralabs/CrisperWhisper2.0_*).

Architecturally identical to Whisper; differs only in the decoder prompt, which
carries mode-control tags before ``<|startoftranscript|>``, and in one unused
training-time head in the checkpoint.
"""

import re
from collections.abc import Iterable

import numpy as np
import torch

from vllm.config import SpeechToTextConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, PromptType, TextPrompt
from vllm.multimodal import MULTIMODAL_REGISTRY

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

_PROMPT_ARTIFACT_RE = re.compile(
    r"\[verbatim_\d+\]|\[intended_\d+\]"
    r"|<ctx>.*?<ectx>|<htx>.*?<ehtx>|<vtx>.*?<evtx>",
    re.DOTALL,
)


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
        if context := (stt_params.request_prompt or "").strip():
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # encoder_blank_head is a training-time alignment head with no
        # inference-time module.
        loader = AutoWeightsLoader(
            self, skip_prefixes=["proj_out.", "encoder_blank_head."]
        )
        weights = _create_fake_bias_for_k_proj(weights, ".k_proj.weight")
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
