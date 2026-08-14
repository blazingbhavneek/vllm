# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Word timings from cross-attention, checked against the reference.

`crisperwhisper` is the upstream package this is ported from. Where it is
installed every stage is compared against it numerically; the rest of the
suite runs regardless.
"""

import numpy as np
import pytest

from vllm.model_executor.models.crisperwhisper_timing import (
    WordTiming,
    align_word_groups,
    blank_logp_from_mel_energy,
    blank_logp_from_space_attention,
    extract_word_timings,
    group_tokens_into_words,
    split_interword_gaps,
    token_logp_from_attention,
    viterbi_align_tokens_with_blanks,
    viterbi_align_words_with_blanks,
)

reference = pytest.importorskip(
    "crisperwhisper.word_timing",
    reason="reference implementation not installed",
)

NUM_TOKENS = 12
NUM_FRAMES = 60
NUM_MELS = 80
SPACE = 220


def _attention(num_tokens: int = NUM_TOKENS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((num_tokens, NUM_FRAMES)) ** 3
    # Give each row a distinct peak so the alignment is monotone and stable.
    for row in range(num_tokens):
        centre = int((row + 0.5) * NUM_FRAMES / num_tokens)
        raw[row, centre] += 5.0
    return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)


def _mel(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mel = rng.normal(size=(NUM_MELS, NUM_FRAMES * 2)).astype(np.float32)
    mel[:, : NUM_FRAMES // 2] -= 6.0  # a quiet lead-in
    return mel


def _sequence() -> tuple[list[int], list[str]]:
    # "la" + "mb" carry no leading space, so they must merge into one word.
    ids = [50258, 100, SPACE, 101, SPACE, 102, SPACE, 103, SPACE, 104, 105, 50257]
    pieces = [
        "<|startoftranscript|>",
        "Mary",
        " ",
        "had",
        " ",
        "a",
        " ",
        "little",
        " ",
        "la",
        "mb",
        "<|endoftext|>",
    ]
    return ids, pieces


def test_token_grouping_matches_reference():
    ids, pieces = _sequence()

    got_indices, got_words = group_tokens_into_words(ids, pieces)
    want_indices, want_words = reference.group_tokens_into_words(ids, pieces)

    assert got_words == want_words
    assert got_indices == want_indices
    # Specials and the space token must not become words.
    assert got_words == ["Mary", "had", "a", "little", "lamb"]


def test_token_grouping_skips_context_and_hotword_spans():
    # A longform continuation prompt is `{mode tags}<ctx> ... <ectx>` (prior
    # text, for the model to condition on) `<htx> ... <ehtx>` (hotwords)
    # `{whisper prefix}{new text}` all as one decoder sequence -- captured
    # attention spans the whole teacher-forced pass, not just the new text,
    # so the words inside <ctx>/<htx> must not leak into the result.
    ids = [
        1,
        2,
        3,
        100,
        SPACE,
        101,
        4,
        5,
        6,
        SPACE,
        107,
        SPACE,
        108,
    ]
    pieces = [
        "<ctx>",
        "prior",
        "words",
        "<ectx>",
        " ",
        "<htx>",
        "hot",
        "word",
        "<ehtx>",
        " ",
        "new",
        " ",
        "text",
    ]

    got_indices, got_words = group_tokens_into_words(ids, pieces)

    assert got_words == ["new", "text"]
    assert got_indices == [[10], [12]]


def test_token_grouping_handles_an_unclosed_span():
    # A truncated or malformed prompt should not spill every following
    # token into the suppressed span forever.
    ids = [1, 100, SPACE, 101]
    pieces = ["<ctx>", "prior", " ", "word"]

    _, got_words = group_tokens_into_words(ids, pieces)

    assert got_words == []


@pytest.mark.parametrize("sharpen", [1.0, 3.0, 5.0])
def test_token_logp_matches_reference(sharpen):
    attention = _attention()

    got = token_logp_from_attention(attention, sharpen=sharpen)
    want = reference.token_logp_from_attention(attention, sharpen=sharpen)

    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(("gamma", "penalty"), [(1.0, 0.0), (3.0, 3.0)])
def test_mel_blank_matches_reference(gamma, penalty):
    mel = _mel()

    got = blank_logp_from_mel_energy(
        mel, target_frames=NUM_FRAMES, gamma=gamma, penalty=penalty
    )
    want = reference.blank_logp_from_mel_energy(
        mel, target_frames=NUM_FRAMES, gamma=gamma, penalty=penalty
    )

    np.testing.assert_array_equal(got, want)


def test_space_blank_matches_reference():
    ids, _ = _sequence()
    attention = _attention()

    got = blank_logp_from_space_attention(attention, ids, target_frames=NUM_FRAMES)
    want = reference.blank_logp_from_space_attention(
        attention, ids, target_frames=NUM_FRAMES
    )

    assert got is not None
    np.testing.assert_array_equal(got, want)


def test_space_blank_is_none_without_space_tokens():
    attention = _attention(num_tokens=3)
    assert blank_logp_from_space_attention(attention, [1, 2, 3], NUM_FRAMES) is None


def test_viterbi_matches_reference():
    attention = _attention()
    token_logp = token_logp_from_attention(attention, sharpen=5.0)
    blank_logp = blank_logp_from_mel_energy(_mel(), target_frames=NUM_FRAMES)

    got = viterbi_align_tokens_with_blanks(token_logp, blank_logp)
    want = reference.viterbi_align_tokens_with_blanks(token_logp, blank_logp)

    assert got == want


def test_word_viterbi_matches_reference():
    ids, pieces = _sequence()
    indices, _ = group_tokens_into_words(ids, pieces)
    token_logp = token_logp_from_attention(_attention(), sharpen=5.0)
    blank_logp = blank_logp_from_mel_energy(_mel(), target_frames=NUM_FRAMES)

    got = viterbi_align_words_with_blanks(token_logp, blank_logp, indices)
    want = reference.viterbi_align_words_with_blanks(token_logp, blank_logp, indices)

    assert got == want


def test_viterbi_rejects_mismatched_blank_length():
    token_logp = token_logp_from_attention(_attention())
    with pytest.raises(ValueError):
        viterbi_align_tokens_with_blanks(token_logp, np.zeros(NUM_FRAMES + 1))


def test_end_to_end_matches_reference():
    """The whole pipeline, against the reference's public entry point."""
    ids, pieces = _sequence()
    attention, mel = _attention(), _mel()

    class _Engine:
        """The reference takes an engine only to decode ids to pieces."""

        class tokenizer:  # noqa: N801
            @staticmethod
            def decode(token_ids):
                return pieces[ids.index(token_ids[0])]

    got = extract_word_timings(ids, pieces, attention, mel)
    want = reference.extract_word_timings(_Engine(), ids, attention, mel)

    assert [w.word for w in got] == [w.word for w in want]
    assert [w.start for w in got] == [w.start for w in want]
    assert [w.end for w in got] == [w.end for w in want]


def test_timings_are_ordered_and_inside_the_window():
    ids, pieces = _sequence()

    words = extract_word_timings(ids, pieces, _attention(), _mel())

    assert words, "expected at least one placed word"
    for word in words:
        assert 0.0 <= word.start <= word.end <= NUM_FRAMES * 0.02
    for left, right in zip(words, words[1:]):
        assert left.end <= right.start


def test_keep_unplaceable_stays_one_to_one_with_words():
    ids, pieces = _sequence()
    _, word_texts = group_tokens_into_words(ids, pieces)

    kept = extract_word_timings(
        ids, pieces, _attention(), _mel(), keep_unplaceable=True
    )

    assert [w.word for w in kept] == word_texts


def test_short_gaps_are_shared_and_real_pauses_are_not():
    words = [
        WordTiming("a", 0.0, 1.0),
        WordTiming("b", 1.04, 2.0),  # 40ms gap -> shared
        WordTiming("c", 3.0, 4.0),  # 1s pause -> untouched
    ]

    split_interword_gaps(words, max_gap_s=0.1)

    assert words[0].end == pytest.approx(1.02)
    assert words[1].start == pytest.approx(1.02)
    assert words[1].end == pytest.approx(2.0)
    assert words[2].start == pytest.approx(3.0)


def test_empty_input_yields_no_words():
    assert extract_word_timings([], [], np.zeros((0, NUM_FRAMES)), _mel()) == []


def test_worker_half_matches_the_full_pipeline():
    """align_word_groups is what runs where the tokenizer is not available."""
    ids, pieces = _sequence()
    attention, mel = _attention(), _mel()
    indices, _ = group_tokens_into_words(ids, pieces)

    spans = align_word_groups(ids, indices, attention, mel)
    full = extract_word_timings(
        ids, pieces, attention, mel, split_gap_max_s=0.0, keep_unplaceable=True
    )

    assert len(spans) == len(full)
    assert [s for s, _ in spans] == [w.start for w in full]
    assert [e for _, e in spans] == [w.end for w in full]


def test_worker_half_needs_no_token_pieces():
    """Only ids and the grouping cross into the worker, never decoded text."""
    ids, pieces = _sequence()
    indices, _ = group_tokens_into_words(ids, pieces)

    spans = align_word_groups(ids, indices, _attention(), _mel())

    assert len(spans) == len(indices)
    assert all(isinstance(s, float) and isinstance(e, float) for s, e in spans)
