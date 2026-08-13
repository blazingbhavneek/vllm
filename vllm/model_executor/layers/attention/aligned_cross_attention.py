# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-attention that can also report its post-softmax weights.

Word-level timestamps for Whisper-family models come from a DTW over the
decoder's cross-attention to the encoder frames, averaged over a handful of
"alignment heads" the checkpoint names in its generation config. Fused
attention kernels never materialize that matrix, so this layer recomputes it
for the selected heads only, next to the normal fused call, which is left
untouched and still produces the output.

Capture is only possible on a pass that carries encoder states, i.e. a single
teacher-forced forward over an already-decoded transcript; on decode steps the
keys live in the paged cache in a backend-specific layout. Restricting to that
pass is what keeps this cheap and CUDA-graph-safe: nothing has to survive
across decode steps or graph replays, and the cost is one `[T, D] x [D, F]`
product per selected head, well under a MFLOP for a 30s window.
"""

import functools
from collections import defaultdict
from collections.abc import Sequence

import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.cross_attention import (
    CrossAttention,
    create_cross_attention_backend,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.selector import get_attn_backend


def cross_attention_probs(
    query: torch.Tensor,
    key: torch.Tensor,
    scale: float,
    head_indices: Sequence[int],
) -> torch.Tensor:
    """Post-softmax cross-attention for selected query heads.

    Args:
        query: Decoder queries, `[num_tokens, num_heads, head_size]`.
        key: Encoder keys, `[num_frames, num_kv_heads, head_size]`.
        scale: Softmax scale the attention layer applies to `q @ k.T`.
        head_indices: Query-head indices to score.

    Returns:
        `[len(head_indices), num_tokens, num_frames]`, float32.

    Raises:
        ValueError: If `head_indices` is empty or out of range.
    """
    if not head_indices:
        raise ValueError("head_indices must not be empty")

    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if any(head < 0 or head >= num_heads for head in head_indices):
        raise ValueError(
            f"head index out of range for {num_heads} heads: {list(head_indices)}"
        )

    # Under GQA/MQA several query heads share one kv head; Whisper is 1:1.
    heads_per_kv = num_heads // num_kv_heads
    kv_indices = [head // heads_per_kv for head in head_indices]

    selected_q = query[:, list(head_indices), :].to(torch.float32)
    selected_k = key[:, kv_indices, :].to(torch.float32)

    scores = torch.einsum("thd,fhd->htf", selected_q, selected_k) * scale
    return torch.softmax(scores, dim=-1)


class CrossAttentionCapture:
    """Collects alignment-head cross-attention across the decoder stack.

    One instance per teacher-forced pass. Layers append to it as they run;
    `averaged` then gives the matrix the DTW consumes.
    """

    def __init__(self, alignment_heads: Sequence[tuple[int, int]]) -> None:
        if not alignment_heads:
            raise ValueError("alignment_heads must not be empty")
        self.heads_by_layer: dict[int, list[int]] = defaultdict(list)
        for layer_index, head_index in alignment_heads:
            self.heads_by_layer[int(layer_index)].append(int(head_index))
        self._captured: list[torch.Tensor] = []

    def wants(self, layer_index: int) -> bool:
        return layer_index in self.heads_by_layer

    def capture(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        scale: float,
    ) -> None:
        self._captured.append(
            cross_attention_probs(
                query, key, scale, self.heads_by_layer[layer_index]
            )
        )

    def averaged(self) -> torch.Tensor | None:
        """Mean over every captured head, `[num_tokens, num_frames]`."""
        if not self._captured:
            return None
        return torch.cat(self._captured, dim=0).mean(dim=0)

    def reset(self) -> None:
        self._captured.clear()


@functools.lru_cache
def create_aligned_cross_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    cross_backend = create_cross_attention_backend(underlying_attn_backend)
    cross_impl = cross_backend.get_impl_cls()

    class AlignedCrossAttentionImpl(cross_impl):  # type: ignore[valid-type,misc]
        def forward(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata,
            output: torch.Tensor,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            out = super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

            capture = getattr(layer, "alignment_capture", None)
            # `key` is None on decode steps, where the encoder keys are only
            # in the paged cache. Those passes are simply not scored.
            if capture is not None and key is not None:
                layer_index = layer.alignment_layer_index
                if capture.wants(layer_index):
                    assert layer.query_quant is None, (
                        "alignment capture needs unquantized queries"
                    )
                    capture.capture(layer_index, query, key, self.scale)

            return out

    return subclass_attention_backend_with_overrides(
        name_prefix="Aligned",
        attention_backend_cls=cross_backend,
        overrides={"get_impl_cls": lambda: AlignedCrossAttentionImpl},
    )


class AlignedCrossAttention(CrossAttention):
    """Cross-attention that can expose post-softmax weights for some heads.

    Behaves exactly like `CrossAttention` until `alignment_capture` is set on
    the layer, which callers do for the duration of one teacher-forced pass.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        cache_config: CacheConfig | None = None,
        attn_type: str | None = None,
        *,
        alignment_layer_index: int,
        **kwargs,
    ):
        if attn_type is not None:
            assert attn_type == AttentionType.ENCODER_DECODER, (
                "AlignedCrossAttention only supports AttentionType.ENCODER_DECODER"
            )

        kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        underlying_attn_backend = get_attn_backend(
            head_size,
            torch.get_default_dtype(),
            kv_cache_dtype,
            attn_type=AttentionType.ENCODER_DECODER,
        )

        # Skips CrossAttention.__init__, which hardcodes the plain cross
        # backend; the kv cache spec is still inherited from it.
        Attention.__init__(
            self,
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            cache_config=cache_config,
            attn_backend=create_aligned_cross_attention_backend(
                underlying_attn_backend
            ),
            attn_type=AttentionType.ENCODER_DECODER,
            **kwargs,
        )

        self.alignment_layer_index = alignment_layer_index
        self.alignment_capture: CrossAttentionCapture | None = None
