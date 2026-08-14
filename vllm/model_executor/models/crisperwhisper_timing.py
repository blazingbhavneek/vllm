# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Word timings from CrisperWhisper cross-attention.

A port of `crisperwhisper/word_timing.py`: given post-softmax cross-attention
`[num_tokens, num_frames]` averaged over the checkpoint's alignment heads, plus
the log-mel of the same window, recover per-word start/end seconds by

1. grouping tokens into words,
2. turning attention into a per-token log-probability over encoder frames,
3. reading a per-frame "blank" (silence) log-probability off the mel energy,
4. running a word-level Viterbi with a virtual blank between every word.

Numpy only, and free of any engine handle, so it runs wherever the attention
was captured and is verifiable against the reference on synthetic input.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

FRAME_DURATION_S = 0.02
SPACE_TOKEN_ID = 220

_PROMPT_TAG_PREFIXES = ("[verbatim_", "[intended_")
_MARKER_PIECES = frozenset(
    {"<ctx>", "<ectx>", "<htx>", "<ehtx>", "<vtx>", "<evtx>", "<sot>", "<eot>"}
)
# The prompt's <ctx>/<htx>/<vtx> spans carry prior-context or hotword text --
# real words, but not part of what this pass generated -- so everything
# between an opener and its closer is skipped, not just the markers.
_SPAN_CLOSERS = {"<ctx>": "<ectx>", "<htx>": "<ehtx>", "<vtx>": "<evtx>"}


@dataclass
class WordTiming:
    """A word and where it falls in the window, in seconds.

    `start`/`end` are `None` only for words the Viterbi could not place, and
    only when the caller asked to keep them.
    """

    word: str
    start: float | None
    end: float | None


def _is_special_token(piece: str) -> bool:
    if piece.startswith("<|") and piece.endswith("|>"):
        return True
    if any(piece.startswith(prefix) for prefix in _PROMPT_TAG_PREFIXES):
        return True
    return piece in _MARKER_PIECES


def _is_space_token(token_id: int, piece: str) -> bool:
    # CrisperWhisper emits an explicit space token; stock Whisper merges the
    # space into the following piece.
    return token_id == SPACE_TOKEN_ID or piece == " "


def _starts_new_word(token_id: int, piece: str) -> bool:
    return _is_space_token(token_id, piece) or piece.startswith(" ")


def group_tokens_into_words(
    token_ids: Sequence[int],
    token_pieces: Sequence[str],
) -> tuple[list[list[int]], list[str]]:
    """Group token positions into words.

    Args:
        token_ids: Generated token ids.
        token_pieces: Each id decoded on its own, positionally matching.

    Returns:
        `(word_token_indices, word_texts)`.
    """
    word_token_indices: list[list[int]] = []
    word_texts: list[str] = []

    current_indices: list[int] = []
    current_text = ""
    span_closer: str | None = None

    def flush() -> None:
        nonlocal current_indices, current_text
        if current_text.strip() and current_indices:
            word_token_indices.append(current_indices)
            word_texts.append(current_text.strip())
        current_indices = []
        current_text = ""

    for index, (token_id, piece) in enumerate(zip(token_ids, token_pieces)):
        if span_closer is not None:
            flush()
            if piece == span_closer:
                span_closer = None
            continue
        if piece in _SPAN_CLOSERS:
            flush()
            span_closer = _SPAN_CLOSERS[piece]
            continue
        if _is_special_token(piece) or _is_space_token(token_id, piece):
            flush()
            continue
        if _starts_new_word(token_id, piece) and current_text != "":
            flush()
        current_indices.append(index)
        current_text += piece

    flush()
    return word_token_indices, word_texts


def token_logp_from_attention(
    attention: np.ndarray,
    sharpen: float = 3.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-token log-probability over encoder frames.

    Args:
        attention: Post-softmax attention, `[num_tokens, num_frames]`.
        sharpen: Exponent applied before renormalizing; `> 1` sharpens peaks.
        eps: Floor guarding the log and the row sums.

    Returns:
        `[num_tokens, num_frames]` float32 log-probabilities.
    """
    attn = attention.astype(np.float64)
    attn[attn < 0] = 0.0
    if sharpen is not None and sharpen != 1.0:
        attn = np.power(attn, sharpen)
    row_sums = np.clip(attn.sum(axis=1, keepdims=True), eps, None)
    return np.log(attn / row_sums + eps).astype(np.float32)


def _resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if x.size == target_len:
        return x.astype(np.float32)
    if x.size <= 1:
        fill = float(x[0]) if x.size == 1 else 0.0
        return np.full((target_len,), fill, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def blank_logp_from_mel_energy(
    mel: np.ndarray,
    target_frames: int,
    audio_frames: int | None = None,
    eps: float = 1e-6,
    gamma: float = 1.0,
    penalty: float = 0.0,
) -> np.ndarray:
    """Per-frame silence log-probability, read off the mel energy.

    Args:
        mel: Log-mel spectrogram `[num_mels, num_mel_frames]`, at twice the
            encoder frame rate.
        target_frames: Encoder frames to resample onto.
        audio_frames: Optional clamp, in encoder frames, so blank statistics
            are not drawn from the silent padding of a short window.
        eps: Floor guarding the log.
        gamma: Exponent on the blank probability; `> 1` makes the blank fire
            only on genuinely silent frames rather than quiet speech.
        penalty: Constant subtracted from the log-probability, a prior against
            entering a blank state without strong silence evidence.

    Returns:
        `[target_frames]` float32 log-probabilities.
    """
    if mel.ndim == 3:
        mel = mel[0]

    if audio_frames is not None:
        mel = mel[:, : min(mel.shape[1], audio_frames * 2)]

    energy = mel.mean(axis=0).astype(np.float32)

    p10 = float(np.percentile(energy, 10))
    p90 = float(np.percentile(energy, 90))
    energy_norm = np.clip((energy - p10) / max(1e-6, p90 - p10), 0.0, 1.0)
    energy_norm = _resample_1d(energy_norm, target_frames)

    blank_p = np.clip(1.0 - energy_norm, 1e-4, 1.0 - 1e-4)
    if gamma != 1.0:
        blank_p = np.clip(blank_p**gamma, 1e-4, 1.0)
    logp = np.log(blank_p + eps).astype(np.float32)
    if penalty:
        logp = (logp - penalty).astype(np.float32)
    return logp


def blank_logp_from_space_attention(
    attention: np.ndarray,
    token_ids: Sequence[int],
    target_frames: int,
    sharpen: float = 3.0,
    eps: float = 1e-8,
) -> np.ndarray | None:
    """Per-frame pause log-probability, from the explicit space token.

    The frames a space token attends to are the inter-word gaps, a cleaner
    pause signal than mel energy where the tokenizer has such a token.

    Returns:
        `[target_frames]` float32 log-probabilities, or `None` when the
        sequence contains no space tokens.
    """
    space_rows = [
        index
        for index, token_id in enumerate(token_ids)
        if int(token_id) == SPACE_TOKEN_ID and index < attention.shape[0]
    ]
    if not space_rows:
        return None

    spaces = attention[space_rows, :target_frames].astype(np.float64)
    spaces[spaces < 0] = 0.0
    if sharpen is not None and sharpen != 1.0:
        spaces = np.power(spaces, sharpen)

    row_sums = np.clip(spaces.sum(axis=1, keepdims=True), eps, None)
    blank = (spaces / row_sums).max(axis=0)

    peak = float(blank.max())
    if peak > 0:
        blank = blank / peak
    blank = np.clip(blank, 1e-4, 1.0 - 1e-4)
    return np.log(blank + eps).astype(np.float32)


def viterbi_align_tokens_with_blanks(
    token_logp: np.ndarray,
    blank_logp: np.ndarray,
    frame_duration: float = FRAME_DURATION_S,
) -> list[tuple[float | None, float | None]]:
    """Align rows to frames through `[blank, row, blank, row, ..., blank]`.

    Args:
        token_logp: Emission log-probabilities, `[num_rows, num_frames]`.
        blank_logp: Shared blank emission, `[num_frames]`.
        frame_duration: Seconds per encoder frame.

    Returns:
        `(start, end)` per row, `(None, None)` where no frame was assigned.

    Raises:
        ValueError: If `blank_logp` does not match the frame count.
    """
    num_rows, num_frames = token_logp.shape
    if num_rows == 0 or num_frames == 0:
        return []
    if blank_logp.shape[0] != num_frames:
        raise ValueError("blank_logp length must match number of frames")

    num_states = 2 * num_rows + 1
    neg_inf = np.float32(-1e9)

    emit = np.empty((num_states, num_frames), dtype=np.float32)
    emit[0::2, :] = blank_logp[np.newaxis, :]
    emit[1::2, :] = token_logp

    dp = np.full((num_states, num_frames), neg_inf, dtype=np.float32)
    back = np.zeros((num_states, num_frames), dtype=np.int8)
    dp[0, 0] = emit[0, 0]

    # Only the time loop stays in Python; each step depends on the last.
    advance = np.full((num_states,), neg_inf, dtype=np.float32)
    for frame in range(1, num_frames):
        stay = dp[:, frame - 1]
        advance[1:] = stay[:-1]
        take_advance = advance > stay
        dp[:, frame] = np.where(take_advance, advance, stay) + emit[:, frame]
        back[:, frame] = take_advance

    # Tie-break toward later states so a trailing blank does not win.
    end_state = int(
        np.argmax(
            dp[:, num_frames - 1] + np.arange(num_states, dtype=np.float32) * 1e-4
        )
    )
    states = np.empty((num_frames,), dtype=np.int32)
    state = end_state
    for frame in range(num_frames - 1, -1, -1):
        states[frame] = state
        if frame == 0:
            break
        if back[state, frame] == 1:
            state -= 1

    spans: list[tuple[float | None, float | None]] = []
    for row in range(num_rows):
        frames = np.where(states == 2 * row + 1)[0]
        if frames.size == 0:
            spans.append((None, None))
        else:
            spans.append(
                (int(frames[0]) * frame_duration, int(frames[-1]) * frame_duration)
            )
    return spans


def viterbi_align_words_with_blanks(
    token_logp: np.ndarray,
    blank_logp: np.ndarray,
    word_token_indices: Sequence[Sequence[int]],
    frame_duration: float = FRAME_DURATION_S,
) -> list[tuple[float | None, float | None]]:
    """As above, after collapsing each word's token rows via logsumexp."""
    num_tokens, num_frames = token_logp.shape
    num_words = len(word_token_indices)
    if num_words == 0:
        return []

    word_logp = np.full((num_words, num_frames), -1e9, dtype=np.float32)
    for word, token_indices in enumerate(word_token_indices):
        valid = [i for i in token_indices if 0 <= i < num_tokens]
        if valid:
            word_logp[word, :] = np.logaddexp.reduce(token_logp[valid, :], axis=0)

    return viterbi_align_tokens_with_blanks(word_logp, blank_logp, frame_duration)


def split_interword_gaps(
    words: list[WordTiming],
    max_gap_s: float = 0.1,
) -> list[WordTiming]:
    """Share short inter-word silences evenly between neighbours.

    The Viterbi parks silence in a blank state, so a word ends at the onset of
    its last emitting frame and the gap to the next word is unassigned. For
    gaps up to `max_gap_s` both boundaries move to the midpoint; genuine pauses
    are left alone. Mutates and returns `words`.
    """
    if max_gap_s <= 0 or len(words) < 2:
        return words
    for left, right in zip(words, words[1:]):
        gap = right.start - left.end
        if 0.0 < gap <= max_gap_s:
            midpoint = float(left.end + gap / 2.0)
            left.end = midpoint
            right.start = midpoint
    return words


def align_word_groups(
    token_ids: Sequence[int],
    word_token_indices: Sequence[Sequence[int]],
    attention: np.ndarray,
    mel: np.ndarray,
    *,
    audio_duration_s: float | None = None,
    sharpen: float = 5.0,
    frame_duration: float = FRAME_DURATION_S,
    blank_source: str = "mel",
    blank_gamma: float = 3.0,
    blank_penalty: float = 3.0,
    clip_to_audio: bool = False,
) -> list[tuple[float | None, float | None]]:
    """Align a precomputed word grouping to frames. No tokenizer needed.

    This is the half that has to run where the attention and the mel are, so
    it deliberately takes the grouping as data. Only `[(start, end), ...]`
    comes back, one entry per word group, which is what crosses a process
    boundary rather than the `[num_tokens, num_frames]` matrix.

    Args:
        token_ids: Generated token ids, 1-to-1 with `attention` rows. Only
            used to locate space tokens for the `"space"` blank source.
        word_token_indices: Token positions per word, from
            `group_tokens_into_words`.
        attention: Post-softmax cross-attention `[num_tokens, num_frames]`,
            averaged over the alignment heads.
        mel: Log-mel of the same window, `[num_mels, num_mel_frames]`.
        audio_duration_s: Window length; only used with `clip_to_audio`.
        sharpen: Attention sharpening exponent.
        frame_duration: Seconds per encoder frame.
        blank_source: `"mel"`, `"space"`, or `"auto"`.
        blank_gamma: Blank exponent, see `blank_logp_from_mel_energy`.
        blank_penalty: Log-prior against blank states.
        clip_to_audio: Restrict analysis to `audio_duration_s`.

    Returns:
        `(start, end)` seconds per word group, `(None, None)` if unplaceable.
    """
    if not word_token_indices or attention.shape[0] == 0:
        return []

    total_frames = attention.shape[1]
    if clip_to_audio and audio_duration_s is not None:
        frames = min(
            total_frames, max(1, int(round(audio_duration_s / frame_duration)))
        )
        mel_audio_frames: int | None = frames
    else:
        frames = total_frames
        mel_audio_frames = None

    attn = attention[:, :frames].astype(np.float32)
    token_logp = token_logp_from_attention(attn, sharpen=sharpen)

    blank_logp = None
    if blank_source in ("auto", "space"):
        blank_logp = blank_logp_from_space_attention(
            attn, token_ids, target_frames=frames, sharpen=sharpen
        )
    if blank_logp is None:
        blank_logp = blank_logp_from_mel_energy(
            mel,
            target_frames=frames,
            audio_frames=mel_audio_frames,
            gamma=blank_gamma,
            penalty=blank_penalty,
        )

    return viterbi_align_words_with_blanks(
        token_logp, blank_logp, word_token_indices, frame_duration=frame_duration
    )


def extract_word_timings(
    token_ids: Sequence[int],
    token_pieces: Sequence[str],
    attention: np.ndarray,
    mel: np.ndarray,
    *,
    audio_duration_s: float | None = None,
    sharpen: float = 5.0,
    frame_duration: float = FRAME_DURATION_S,
    blank_source: str = "mel",
    blank_gamma: float = 3.0,
    blank_penalty: float = 3.0,
    clip_to_audio: bool = False,
    split_gap_max_s: float = 0.1,
    keep_unplaceable: bool = False,
) -> list[WordTiming]:
    """Word timings for one window, in window-local seconds.

    Defaults mirror the reference. `sharpen` and the blank shaping move
    together: a softer `sharpen` with this shaping over-extends words.

    Args:
        token_ids: Generated token ids, 1-to-1 with `attention` rows.
        token_pieces: Each id decoded on its own.
        attention: Post-softmax cross-attention `[num_tokens, num_frames]`,
            already averaged over the alignment heads.
        mel: Log-mel of the same window, `[num_mels, num_mel_frames]`.
        audio_duration_s: Window length; only used with `clip_to_audio`.
        sharpen: Attention sharpening exponent.
        frame_duration: Seconds per encoder frame.
        blank_source: `"mel"`, `"space"`, or `"auto"`. `"auto"` and `"space"`
            prefer the space token and fall back to mel energy.
        blank_gamma: Blank exponent, see `blank_logp_from_mel_energy`.
        blank_penalty: Log-prior against blank states.
        clip_to_audio: Restrict analysis to `audio_duration_s`. Off by
            default: the full window lets blank states absorb the padding,
            and clipping re-derives the energy percentiles over speech only,
            which collapses words onto their attention peak.
        split_gap_max_s: Gaps up to this are shared between neighbours; `0`
            disables.
        keep_unplaceable: Keep unplaceable words as `None`-timed placeholders
            so the result stays 1-to-1 with the word segmentation.

    Returns:
        `WordTiming` per word, in order.
    """
    if len(token_ids) == 0 or attention.shape[0] == 0:
        return []

    word_token_indices, word_texts = group_tokens_into_words(token_ids, token_pieces)
    if not word_token_indices:
        return []

    spans = align_word_groups(
        token_ids,
        word_token_indices,
        attention,
        mel,
        audio_duration_s=audio_duration_s,
        sharpen=sharpen,
        frame_duration=frame_duration,
        blank_source=blank_source,
        blank_gamma=blank_gamma,
        blank_penalty=blank_penalty,
        clip_to_audio=clip_to_audio,
    )

    placed: list[WordTiming] = []
    placed_indices: list[int] = []
    for index, (word, (start, end)) in enumerate(zip(word_texts, spans)):
        if start is None or end is None:
            continue
        placed.append(WordTiming(word=word, start=float(start), end=float(end)))
        placed_indices.append(index)

    if split_gap_max_s and split_gap_max_s > 0:
        placed = split_interword_gaps(placed, max_gap_s=split_gap_max_s)

    if not keep_unplaceable:
        return placed

    full = [WordTiming(word=word, start=None, end=None) for word in word_texts]
    for timing, index in zip(placed, placed_indices):
        full[index] = timing
    return full
