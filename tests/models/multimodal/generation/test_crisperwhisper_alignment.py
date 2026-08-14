# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alignment-head cross-attention capture for CrisperWhisper, on CPU.

Covers the scoring and collection. Wiring the decoder stack to a backend
needs a GPU and is exercised by the model tests.
"""

import numpy as np
import pytest
import torch

from vllm.model_executor.models.crisperwhisper import (
    WORD_TIMING_CAPABILITY,
    CrisperWhisperCrossAttention,
    CrisperWhisperDecoder,
    CrisperWhisperDecoderLayer,
    CrossAttentionCapture,
    RequestSlice,
    batch_request_slices,
    capture_key,
    cross_attention_probs,
)
from vllm.model_executor.models.whisper import (
    WhisperCrossAttention,
    WhisperDecoderLayer,
)

NUM_TOKENS = 7
NUM_FRAMES = 23
NUM_HEADS = 6
HEAD_SIZE = 8
SCALE = HEAD_SIZE**-0.5


def _qk(num_kv_heads: int = NUM_HEADS) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, generator=generator)
    key = torch.randn(NUM_FRAMES, num_kv_heads, HEAD_SIZE, generator=generator)
    return query, key


def test_probs_match_a_plain_softmax_reference():
    query, key = _qk()
    head = 3

    got = cross_attention_probs(query, key, SCALE, [head])

    want = torch.softmax(query[:, head] @ key[:, head].T * SCALE, dim=-1)
    assert got.shape == (1, NUM_TOKENS, NUM_FRAMES)
    torch.testing.assert_close(got[0], want)


def test_probs_are_a_distribution_over_frames():
    query, key = _qk()

    probs = cross_attention_probs(query, key, SCALE, [0, 2, 5])

    assert probs.shape == (3, NUM_TOKENS, NUM_FRAMES)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(3, NUM_TOKENS))


def test_head_order_is_preserved():
    query, key = _qk()

    both = cross_attention_probs(query, key, SCALE, [4, 1])

    only_4 = cross_attention_probs(query, key, SCALE, [4])
    only_1 = cross_attention_probs(query, key, SCALE, [1])
    torch.testing.assert_close(both[0], only_4[0])
    torch.testing.assert_close(both[1], only_1[0])


def test_grouped_query_heads_share_a_kv_head():
    """Two query heads mapping to one kv head must read the same keys."""
    query, key = _qk(num_kv_heads=NUM_HEADS // 2)

    probs = cross_attention_probs(query, key, SCALE, [0, 1])

    want = torch.softmax(query[:, 1] @ key[:, 0].T * SCALE, dim=-1)
    torch.testing.assert_close(probs[1], want)


@pytest.mark.parametrize("heads", [[], [NUM_HEADS], [-1]])
def test_invalid_head_selection_is_rejected(heads):
    query, key = _qk()
    with pytest.raises(ValueError):
        cross_attention_probs(query, key, SCALE, heads)


def _whole_batch(capture: CrossAttentionCapture) -> None:
    capture.begin([RequestSlice("only", (0, NUM_TOKENS), (0, NUM_FRAMES))])


def test_capture_averages_every_selected_head_across_layers():
    query, key = _qk()
    capture = CrossAttentionCapture([(0, 1), (0, 4), (3, 2)])
    _whole_batch(capture)

    assert capture.wants(0) and capture.wants(3)
    assert not capture.wants(1)

    capture.capture(0, query, key, SCALE)
    capture.capture(3, query, key, SCALE)
    averaged = capture.averaged("only")

    want = cross_attention_probs(query, key, SCALE, [1, 4, 2]).mean(dim=0)
    assert averaged.shape == (NUM_TOKENS, NUM_FRAMES)
    torch.testing.assert_close(averaged, want)
    # An average of distributions is still a distribution.
    torch.testing.assert_close(averaged.sum(dim=-1), torch.ones(NUM_TOKENS))


def test_capture_is_empty_until_a_layer_runs():
    capture = CrossAttentionCapture([(0, 1)])
    _whole_batch(capture)
    assert capture.averaged("only") is None

    query, key = _qk()
    capture.capture(0, query, key, SCALE)
    assert capture.averaged("only") is not None

    capture.reset()
    assert capture.averaged("only") is None


def test_capture_requires_alignment_heads():
    with pytest.raises(ValueError):
        CrossAttentionCapture([])


def test_decoder_stack_subclasses_whisper_without_touching_it():
    """The capture path must be additive, not a fork of the Whisper stack."""
    assert issubclass(CrisperWhisperCrossAttention, WhisperCrossAttention)
    assert issubclass(CrisperWhisperDecoderLayer, WhisperDecoderLayer)
    # forward is inherited from Whisper: only __init__ swaps the attention.
    assert CrisperWhisperDecoderLayer.forward is WhisperDecoderLayer.forward


def test_decoder_keeps_torch_compile():
    """Subclassing must not silently drop vLLM's compilation."""
    bases = [cls.__name__ for cls in CrisperWhisperDecoder.__mro__]
    assert "TorchCompileWithNoGuardsWrapper" in bases


def test_capture_is_off_by_default():
    """A layer with no capture set must behave exactly like Whisper's."""
    assert CrisperWhisperCrossAttention.__init__ is not WhisperCrossAttention.__init__
    # The only per-instance additions are the capture slot and layer index.
    added = {"alignment_capture", "alignment_layer_index"}
    source = CrisperWhisperCrossAttention.__init__.__code__.co_names
    assert added.issubset(set(source))


def _slice_ids(count: int) -> torch.Tensor:
    return torch.arange(count)


def test_batch_slices_skip_requests_whose_encoder_is_cached():
    """A decode-only request contributes no encoder frames, so it takes none."""
    slices = batch_request_slices(
        _slice_ids(20), np.array([0, 8, 20]), np.array([1500, 0])
    )

    assert len(slices) == 1
    assert slices[0].query == (0, 8)
    assert slices[0].frames == (0, 1500)


def test_batch_slices_give_each_request_its_own_frames():
    slices = batch_request_slices(
        _slice_ids(20), np.array([0, 8, 20]), np.array([1500, 1500])
    )

    assert [s.query for s in slices] == [(0, 8), (8, 20)]
    assert [s.frames for s in slices] == [(0, 1500), (1500, 3000)]
    assert slices[0].key_id != slices[1].key_id


def test_capture_key_depends_only_on_tokens():
    assert capture_key([1, 2, 3]) == capture_key([1, 2, 3])
    assert capture_key([1, 2, 3]) != capture_key([1, 2, 4])


def test_capture_keeps_requests_apart():
    """Scoring must never mix one request's queries with another's frames."""
    generator = torch.Generator().manual_seed(0)
    query = torch.randn(10, NUM_HEADS, HEAD_SIZE, generator=generator)
    key = torch.randn(2 * NUM_FRAMES, NUM_HEADS, HEAD_SIZE, generator=generator)

    capture = CrossAttentionCapture([(0, 1)])
    first = RequestSlice("first", (0, 4), (0, NUM_FRAMES))
    second = RequestSlice("second", (4, 10), (NUM_FRAMES, 2 * NUM_FRAMES))
    capture.begin([first, second])
    capture.capture(0, query, key, SCALE)

    got_first = capture.averaged("first")
    got_second = capture.averaged("second")
    assert got_first.shape == (4, NUM_FRAMES)
    assert got_second.shape == (6, NUM_FRAMES)

    want_first = cross_attention_probs(
        query[0:4], key[0:NUM_FRAMES], SCALE, [1]
    ).mean(dim=0)
    torch.testing.assert_close(got_first, want_first)


def test_capture_drops_slices_that_do_not_fit_the_tensors():
    query = torch.randn(4, NUM_HEADS, HEAD_SIZE)
    key = torch.randn(NUM_FRAMES, NUM_HEADS, HEAD_SIZE)

    capture = CrossAttentionCapture([(0, 1)])
    capture.begin([RequestSlice("past-the-end", (0, 99), (0, NUM_FRAMES))])
    capture.capture(0, query, key, SCALE)

    assert capture.averaged("past-the-end") is None


def test_capture_pairs_mels_with_requests_in_order():
    capture = CrossAttentionCapture([(0, 1)])
    mels = [np.full((2, 4), 1.0), np.full((2, 4), 2.0)]

    capture.offer_mels(mels)
    capture.begin(
        [RequestSlice("a", (0, 2), (0, 10)), RequestSlice("b", (2, 4), (10, 20))]
    )

    np.testing.assert_array_equal(capture.mel_for("a"), mels[0])
    np.testing.assert_array_equal(capture.mel_for("b"), mels[1])


def test_capture_store_is_bounded():
    capture = CrossAttentionCapture([(0, 1)], max_entries=2)
    for index in range(5):
        capture.offer_mels([np.zeros((2, 2))])
        capture.begin([RequestSlice(f"k{index}", (0, 1), (0, 1))])

    assert capture.mel_for("k0") is None
    assert capture.mel_for("k4") is not None


def test_pop_frees_the_capture():
    query = torch.randn(4, NUM_HEADS, HEAD_SIZE)
    key = torch.randn(NUM_FRAMES, NUM_HEADS, HEAD_SIZE)
    capture = CrossAttentionCapture([(0, 1)])
    capture.offer_mels([np.zeros((2, 4))])
    capture.begin([RequestSlice("k", (0, 4), (0, NUM_FRAMES))])
    capture.capture(0, query, key, SCALE)

    attention, mel = capture.pop("k")
    assert attention is not None and mel is not None
    assert capture.pop("k") == (None, None)


def test_rpc_entry_ignores_other_models():
    """collective_rpc reaches every worker, including non-CrisperWhisper ones."""
    from vllm.v1.worker.worker_base import WorkerBase

    class _PlainModelWorker:
        get_model = staticmethod(lambda: object())
        call_model_capability = WorkerBase.call_model_capability

    worker = _PlainModelWorker()
    assert worker.call_model_capability(WORD_TIMING_CAPABILITY, ("k", [[0]], [1])) is None
