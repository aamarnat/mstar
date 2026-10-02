"""Call contract for AMD AITER dense FMHA on Cosmos3 Q/K/V.

This module does not import torch or AITER. The attention manager applies
the decisions here to real tensors. CPU tests load this file directly.

Assumption (the installed ``aiter`` package is not present in this
worktree, so the signature is taken from the call site, not from a local
stub):

vLLM-Omni's ROCm FLASH_ATTN path imports ``aiter.flash_attn_func`` in
``vllm_omni/diffusion/attention/backends/utils/fa.py`` and calls it from
``FlashAttentionImpl.forward_cuda`` as::

    flash_attn_func(query, key, value, causal=..., softmax_scale=...)

That is the dense path. The ring helper in ``ring/ring_kernels.py`` passes
extra arguments (``dropout_p``, ``window_size``, ``alibi_slopes``,
``return_lse``) because it needs the LSE. The ROCm diffusion forward does
not. This backend matches the diffusion forward: those extra names are
not passed. FA2, which AITER's ``softmax_lse`` layout matches, defaults
dropout to 0 and ``window_size`` to ``(-1, -1)`` (no sliding window).

Tensor rank at that call site is FA2 BSHD, not M*'s packed token layout:

    q: [B, Sq, Hq, D]
    k: [B, Sk, Hkv, D]
    v: [B, Sk, Hkv, D]
    out: [B, Sq, Hq, D]  (or ``(out, lse)``; the manager keeps ``out``)

GQA is native. ``Hq`` may differ from ``Hkv``. KV is not repeated.

Cosmos3 action attention has no additive mask:

* Understanding prefill is square causal self-attention over the text
  tokens (``SDPA(is_causal=True)``). ``flash_attn_func(..., causal=True)``
  with ``Sq == Sk`` is the same mask.
* Generation denoise is full attention of the gen queries over
  ``cat([text K/V, gen K/V])`` (``SDPA(is_causal=False)``).
  ``flash_attn_func(..., causal=False)`` attends every query to every key.
  Scale is ``head_dim ** -0.5``, the SDPA default.

Packed batches are one ``flash_attn_func`` call per segment, each with
batch size 1. That keeps request and guidance-branch boundaries without a
padding mask and without ``flash_attn_varlen_func``. The 1-GPU Cosmos3
action step is a single segment.
"""

from __future__ import annotations

# Keyword names of the verified ROCm dense call. Not dropout / window / lse.
AITER_FLASH_ATTN_KWARGS = ("causal", "softmax_scale")

# Cosmos3 Nano action heads. The manager accepts any GQA multiple; these
# are the Nano action shapes.
COSMOS3_Q_HEADS = 32
COSMOS3_KV_HEADS = 8
COSMOS3_HEAD_DIM = 128


def softmax_scale(head_dim: int) -> float:
    """SDPA's default scale, passed explicitly as ``softmax_scale``."""
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    return head_dim ** -0.5


def flash_attn_func_kwargs(*, causal: bool, head_dim: int) -> dict[str, bool | float]:
    """The exact keyword arguments of the verified ROCm ``flash_attn_func`` call."""
    kwargs: dict[str, bool | float] = {
        "causal": bool(causal),
        "softmax_scale": softmax_scale(head_dim),
    }
    if tuple(kwargs) != AITER_FLASH_ATTN_KWARGS:
        raise RuntimeError(f"AITER kwargs drifted: {tuple(kwargs)}")
    return kwargs


def packed_to_bshd_shape(packed: tuple[int, ...]) -> tuple[int, int, int, int]:
    """M* ``[tokens, heads, dim]`` -> AITER ``[1, tokens, heads, dim]``."""
    if len(packed) != 3:
        raise ValueError(
            "M* attention tensors are [tokens, heads, dim], "
            f"got rank {len(packed)} shape {packed}"
        )
    tokens, heads, dim = packed
    if tokens < 0 or heads <= 0 or dim <= 0:
        raise ValueError(f"invalid packed attention shape {packed}")
    return (1, int(tokens), int(heads), int(dim))


def assert_gqa_passthrough(
    q_bshd: tuple[int, int, int, int],
    k_bshd: tuple[int, int, int, int],
    v_bshd: tuple[int, int, int, int],
) -> None:
    """KV heads stay at Hkv. Repeating them out to Hq would change the kernel."""
    bq, sq, hq, dq = q_bshd
    bk, sk, hk, dk = k_bshd
    bv, sv, hv, dv = v_bshd
    if (bq, bk, bv) != (1, 1, 1):
        raise ValueError(
            f"aiter_fmha calls flash_attn_func per segment with batch 1, got {(bq, bk, bv)}"
        )
    if sk != sv or hk != hv or dk != dv:
        raise ValueError(f"K/V layout mismatch {k_bshd} vs {v_bshd}")
    if dq != dk:
        raise ValueError(f"Q/K head dim mismatch {dq} vs {dk}")
    if hq % hk != 0:
        raise ValueError(f"GQA requires Hq ({hq}) to be a multiple of Hkv ({hk})")
    # Hkv is whatever the caller passed. Callers must not tile KV up to Hq
    # before this check; Cosmos3 Nano passes 8 KV heads against 32 query heads.
    if sq < 0 or sk < sq:
        raise ValueError(f"key sequence {sk} is shorter than query sequence {sq}")


def denoise_key_len(prefix_len: int, q_len: int) -> int:
    """Gen queries attend over the frozen text prefix plus the fresh gen tokens."""
    if prefix_len < 0 or q_len < 0:
        raise ValueError(f"negative lengths prefix={prefix_len} q={q_len}")
    return prefix_len + q_len


def step_causal(*, attention_causal: bool, q_len: int, k_len: int) -> bool:
    """Map ``AttentionStep.causal`` onto ``flash_attn_func``'s ``causal`` flag.

    Square causal (understanding prefill) matches SDPA ``is_causal=True``.
    Non-causal with ``k_len >= q_len`` (generation denoise over
    ``[text | gen]``) matches SDPA ``is_causal=False``. A causal step whose
    query block is shorter than the key block uses FA's bottom-right causal
    alignment, which is still ``causal=True`` and is not the Cosmos3 denoise
    mask.
    """
    if q_len < 0 or k_len < q_len:
        raise ValueError(f"bad causal lengths q={q_len} k={k_len}")
    if attention_causal and q_len != k_len:
        return True
    return bool(attention_causal)
