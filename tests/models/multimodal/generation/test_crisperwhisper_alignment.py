# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alignment-head cross-attention capture for CrisperWhisper, on CPU.

Covers the scoring and collection. Wiring the decoder stack to a backend
needs a GPU and is exercised by the model tests.
"""

import pytest
import torch

from vllm.model_executor.models.crisperwhisper import (
    CrisperWhisperCrossAttention,
    CrisperWhisperDecoder,
    CrisperWhisperDecoderLayer,
    CrossAttentionCapture,
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


def test_capture_averages_every_selected_head_across_layers():
    query, key = _qk()
    capture = CrossAttentionCapture([(0, 1), (0, 4), (3, 2)])

    assert capture.wants(0) and capture.wants(3)
    assert not capture.wants(1)

    capture.capture(0, query, key, SCALE)
    capture.capture(3, query, key, SCALE)
    averaged = capture.averaged()

    want = cross_attention_probs(query, key, SCALE, [1, 4, 2]).mean(dim=0)
    assert averaged.shape == (NUM_TOKENS, NUM_FRAMES)
    torch.testing.assert_close(averaged, want)
    # An average of distributions is still a distribution.
    torch.testing.assert_close(averaged.sum(dim=-1), torch.ones(NUM_TOKENS))


def test_capture_is_empty_until_a_layer_runs():
    capture = CrossAttentionCapture([(0, 1)])
    assert capture.averaged() is None

    query, key = _qk()
    capture.capture(0, query, key, SCALE)
    assert capture.averaged() is not None

    capture.reset()
    assert capture.averaged() is None


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
