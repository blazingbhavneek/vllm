# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Longform chunk-boundary decisions: the timestamp-aware trim and the
retry-on-collapse heuristic, both mirroring reference defaults
(`timestamp_aware_drop=True`, `hallucination_mitigation=True`).

These are pure classmethods on `CrisperWhisperForConditionalGeneration` with
no model weights involved, so they are exercised directly rather than
through a live decode.
"""

import numpy as np
import pytest

from vllm.config.speech_to_text import SpeechToTextConfig, SpeechToTextParams
from vllm.model_executor.models.crisperwhisper import (
    EARLY_EOT_CONFIDENT_PROB,
    EARLY_EOT_STOP_PROB_THRESHOLD,
    EARLY_EOT_TAIL_MIN_FINAL_S,
    EARLY_EOT_TAIL_MIN_NONFINAL_S,
    FALLBACK_TEMPERATURES,
    LONGFORM_DROP_WORDS,
    LONGFORM_STRIDE_S,
    MAX_CONSECUTIVE_REPEATS,
    CrisperWhisperForConditionalGeneration,
    _longest_consecutive_run,
)

MODEL = CrisperWhisperForConditionalGeneration


def _params(
    *, max_audio_clip_s: int = 30, language: str | None = "en", speech_s: float = 30.0
) -> SpeechToTextParams:
    sample_rate = 16_000
    # `_speech_active_seconds` gates on the spread between energy
    # percentiles, so constant amplitude reads as silence (zero spread).
    # Fluctuating noise gives every frame a slightly different energy,
    # which the gate reads as speech-active throughout.
    audio = np.random.RandomState(0).uniform(
        -0.1, 0.1, int(speech_s * sample_rate)
    ).astype(np.float32)
    return SpeechToTextParams(
        audio=audio,
        stt_config=SpeechToTextConfig(max_audio_clip_s=max_audio_clip_s),
        model_config=None,  # unused by the hooks under test
        language=language,
    )


def _timings(words: list[str], starts: list[float]) -> list[dict]:
    return [
        {"word": w, "start": s, "end": s + 0.2} for w, s in zip(words, starts)
    ]


# ---------------------------------------------------------------------------
# _longest_consecutive_run
# ---------------------------------------------------------------------------


def test_longest_run_of_empty_is_zero():
    assert _longest_consecutive_run([]) == 0


def test_longest_run_with_no_repeats_is_one():
    assert _longest_consecutive_run(["a", "b", "c"]) == 1


def test_longest_run_finds_a_run_anywhere():
    assert _longest_consecutive_run(["a", "b", "b", "b", "c", "d", "d"]) == 3


# ---------------------------------------------------------------------------
# trim_chunk_boundary
# ---------------------------------------------------------------------------


def test_trim_falls_back_to_fixed_drop_without_timings():
    text = "one two three four five"
    kept = MODEL.trim_chunk_boundary(text, _params())
    assert kept == "one two three"  # 5 words, minus LONGFORM_DROP_WORDS=2


def test_trim_falls_back_when_timings_length_mismatches_words():
    text = "one two three four five"
    mismatched = _timings(["one", "two", "three"], [0.0, 1.0, 2.0])
    kept = MODEL.trim_chunk_boundary(text, _params(), word_timings=mismatched)
    assert kept == "one two three"


def test_trim_keeps_everything_when_no_word_starts_in_the_overlap():
    # 30s window, 26s stride -> overlap starts at 4s. Every word here
    # finished well before that.
    text = "one two three four five"
    words = text.split()
    timings = _timings(words, [0.1, 0.5, 0.9, 1.3, 1.7])
    kept = MODEL.trim_chunk_boundary(text, _params(), word_timings=timings)
    assert kept == text


def test_trim_drops_only_words_that_start_in_the_overlap():
    overlap_start = 30 - LONGFORM_STRIDE_S  # 4.0s
    text = "one two three four five"
    words = text.split()
    # Last word starts inside the overlap; the rest are well before it.
    timings = _timings(words, [0.1, 0.5, 0.9, 1.3, overlap_start + 0.5])
    kept = MODEL.trim_chunk_boundary(text, _params(), word_timings=timings)
    assert kept == "one two three four"


def test_trim_caps_the_drop_at_longform_drop_words():
    overlap_start = 30 - LONGFORM_STRIDE_S
    # Every word starts inside the overlap; only the cap should be dropped.
    words = [f"w{i}" for i in range(6)]
    text = " ".join(words)
    timings = _timings(words, [overlap_start + i for i in range(6)])
    kept = MODEL.trim_chunk_boundary(text, _params(), word_timings=timings)
    assert kept == " ".join(words[: -LONGFORM_DROP_WORDS])
    assert len(words) - len(kept.split()) == LONGFORM_DROP_WORDS


def test_trim_respects_a_non_default_window_duration():
    max_audio_clip_s = 40
    overlap_start = max_audio_clip_s - LONGFORM_STRIDE_S  # 14.0s here, not 4.0s
    text = "one two three four five"
    words = text.split()
    timings = _timings(words, [0.1, 0.5, 0.9, 1.3, overlap_start + 0.5])
    kept = MODEL.trim_chunk_boundary(
        text, _params(max_audio_clip_s=max_audio_clip_s), word_timings=timings
    )
    assert kept == "one two three four"


def test_trim_ignores_timings_for_no_space_languages():
    params = _params(language="ja")
    text = "一二三四五"
    timings = _timings(list(text), [10.0] * len(text))
    assert MODEL.trim_chunk_boundary(text, params, word_timings=timings) == text


def test_trim_is_a_no_op_at_or_below_the_drop_cap():
    text = "one two"
    assert MODEL.trim_chunk_boundary(text, _params()) == text


# ---------------------------------------------------------------------------
# should_retry_chunk
# ---------------------------------------------------------------------------


def test_retry_fires_on_near_empty_collapse():
    text = "meanwhile."
    overrides = MODEL.should_retry_chunk(text, _params(speech_s=30.0), attempt=0)
    assert overrides == {"temperature": FALLBACK_TEMPERATURES[0]}


def test_retry_fires_on_a_runaway_repeat_even_with_plenty_of_words():
    # Enough total words to clear the near-empty gate on its own; the
    # repeat is the only thing wrong with this chunk.
    words = ["hello"] + ["[breath]"] * MAX_CONSECUTIVE_REPEATS + ["world"] * 10
    text = " ".join(words)
    overrides = MODEL.should_retry_chunk(text, _params(speech_s=30.0), attempt=1)
    assert overrides == {"temperature": FALLBACK_TEMPERATURES[1]}


def test_retry_does_not_fire_on_a_short_natural_repeat():
    # Comfortably clears the near-empty gate on its own (30s of speech
    # wants >= 15 words), so only the repeat length is under test.
    filler = [f"word{i}" for i in range(20)]
    words = ["okay", "okay", "okay", "okay"] + filler
    text = " ".join(words)
    assert MODEL.should_retry_chunk(text, _params(speech_s=30.0), attempt=0) is None


def test_retry_is_none_for_a_normal_transcript():
    text = " ".join(f"word{i}" for i in range(20))
    assert MODEL.should_retry_chunk(text, _params(speech_s=30.0), attempt=0) is None


def test_retry_stops_after_the_last_fallback_temperature():
    text = "meanwhile."
    last = len(FALLBACK_TEMPERATURES) - 1
    assert MODEL.should_retry_chunk(text, _params(), attempt=last) is not None
    assert MODEL.should_retry_chunk(text, _params(), attempt=last + 1) is None


def test_retry_ignores_no_space_languages():
    # Would trip the repeat gate under a space-delimited language; the
    # no-space guard must short-circuit before that check runs at all.
    params = _params(language="ja")
    text = " ".join(["あ"] * (MAX_CONSECUTIVE_REPEATS + 5))
    assert MODEL.should_retry_chunk(text, params, attempt=0) is None


# ---------------------------------------------------------------------------
# should_attempt_early_eot_recovery / is_eot_stop_confident /
# is_recovery_confident
# ---------------------------------------------------------------------------


def _noisy_audio(duration_s: float, sample_rate: int = 16_000) -> np.ndarray:
    return np.random.RandomState(1).uniform(
        -0.1, 0.1, int(duration_s * sample_rate)
    ).astype(np.float32)


def test_eot_gap_needs_a_placed_word():
    chunk = _noisy_audio(10.0)
    assert MODEL.should_attempt_early_eot_recovery(
        chunk, _params(), [{"word": "x", "start": 0.0, "end": None}], is_last=False,
    ) is False
    assert MODEL.should_attempt_early_eot_recovery(
        chunk, _params(), [], is_last=False,
    ) is False


def test_eot_gap_fires_when_speech_remains_after_the_last_word():
    chunk = _noisy_audio(10.0)
    timings = [{"word": "hi", "start": 0.0, "end": 2.0}]
    assert MODEL.should_attempt_early_eot_recovery(
        chunk, _params(), timings, is_last=False,
    ) is True  # 8s of noise left, well past the 4.0s non-final floor


def test_eot_gap_does_not_fire_when_nothing_is_left_to_hear():
    chunk = _noisy_audio(10.0)
    timings = [{"word": "hi", "start": 0.0, "end": 9.5}]
    assert MODEL.should_attempt_early_eot_recovery(
        chunk, _params(), timings, is_last=False,
    ) is False  # 0.5s left, under the 4.0s non-final floor


def test_eot_gap_uses_the_smaller_final_floor():
    chunk = _noisy_audio(10.0)
    # 7s left clears the 2.0s final floor but not the 4.0s non-final one.
    timings = [{"word": "hi", "start": 0.0, "end": 3.0}]
    assert MODEL.should_attempt_early_eot_recovery(
        chunk, _params(), timings, is_last=True,
    ) is True
    assert EARLY_EOT_TAIL_MIN_FINAL_S < EARLY_EOT_TAIL_MIN_NONFINAL_S


def test_eot_gap_ignores_no_space_languages():
    chunk = _noisy_audio(10.0)
    params = _params(language="ja")
    timings = [{"word": "hi", "start": 0.0, "end": 1.0}]
    assert MODEL.should_attempt_early_eot_recovery(
        chunk, params, timings, is_last=False,
    ) is False


def test_eot_stop_confidence_threshold():
    assert MODEL.is_eot_stop_confident(EARLY_EOT_STOP_PROB_THRESHOLD) is True
    assert MODEL.is_eot_stop_confident(EARLY_EOT_STOP_PROB_THRESHOLD - 0.01) is False


def test_recovery_confidence_threshold():
    assert MODEL.is_recovery_confident(EARLY_EOT_CONFIDENT_PROB) is True
    assert MODEL.is_recovery_confident(EARLY_EOT_CONFIDENT_PROB - 0.01) is False


def test_early_eot_thresholds_match_the_reference_defaults():
    # EarlyEotConfig in crisperwhisper/longform/base.py.
    assert EARLY_EOT_STOP_PROB_THRESHOLD == 0.7
    assert EARLY_EOT_CONFIDENT_PROB == 0.9
    assert EARLY_EOT_TAIL_MIN_FINAL_S == 2.0
    assert EARLY_EOT_TAIL_MIN_NONFINAL_S == 4.0


def test_early_eot_recovery_is_off_by_default():
    from vllm.model_executor.models.interfaces import SupportsTranscription
    from vllm.model_executor.models.whisper import WhisperForConditionalGeneration

    assert MODEL.supports_early_eot_recovery is True
    assert WhisperForConditionalGeneration.supports_early_eot_recovery is False
    assert SupportsTranscription.supports_early_eot_recovery is False
