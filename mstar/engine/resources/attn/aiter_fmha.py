"""Cosmos3 attention via AMD AITER ``flash_attn_func``, with no paged KV.

Eager only. Understanding prefill and generation denoise both run here when
``attention_backend="aiter_fmha"``. The text prefix is kept in dense buffers
owned by this manager. ``requires_kv_write`` is false, so the layer never
calls ``KVManager.write_kv`` and never scatters K/V into page planes
(``_kv_scatter_nhd_eager`` / ``aten::index_put``). This module does not
import FlashInfer and does not call ``fa3_fwd``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from mstar.engine.resources.attn.aiter_fmha_layout import (
    assert_gqa_passthrough,
    denoise_key_len,
    flash_attn_func_kwargs,
    packed_to_bshd_shape,
    step_causal,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVConfig
from mstar.engine.resources.kv.plan import KVPlanOutput, KVPlanOutputs
from mstar.engine.resources.step import StepContext

logger = logging.getLogger(__name__)


def _unwrap_flash_output(out: torch.Tensor | tuple) -> torch.Tensor:
    # Some FA builds return (out, lse). The verified diffusion call keeps out.
    if isinstance(out, tuple):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"aiter.flash_attn_func returned {type(out).__name__}, expected a Tensor")
    return out


def aiter_flash_attn_func(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    v_bshd: torch.Tensor,
    *,
    causal: bool,
    head_dim: int,
) -> torch.Tensor:
    """One dense AITER FMHA call.

    Ranks are BSHD ``[1, S, H, D]``. Keyword names are ``causal`` and
    ``softmax_scale`` only — see ``aiter_fmha_layout``.
    """
    from aiter import flash_attn_func

    assert_gqa_passthrough(tuple(q_bshd.shape), tuple(k_bshd.shape), tuple(v_bshd.shape))
    kwargs = flash_attn_func_kwargs(causal=causal, head_dim=head_dim)
    out = flash_attn_func(q_bshd, k_bshd, v_bshd, **kwargs)
    return _unwrap_flash_output(out)


def _to_bshd(packed: torch.Tensor) -> torch.Tensor:
    """``[tokens, heads, dim]`` -> contiguous ``[1, tokens, heads, dim]``."""
    packed_to_bshd_shape(tuple(packed.shape))
    return packed.unsqueeze(0).contiguous()


@dataclass(frozen=True)
class _Segment:
    request_id: str
    label: str
    prefix_len: int
    q_len: int
    generation: int


@dataclass(frozen=True)
class _Plan:
    segments: tuple[_Segment, ...]
    causal: bool


@dataclass
class _Prefix:
    generation: int
    k: torch.Tensor  # [prefix_len, Hkv, D]
    v: torch.Tensor


class AiterFmhaAttentionManager(AttentionManager):
    """Dense AITER FMHA for one Cosmos3 attention resource.

    Prefill (``causal=True``, ``prefix_len=0``) runs square causal attention
    on the text tokens and stores that K/V per ``(request, label, layer)``.
    Denoise (``causal=False``) concatenates the stored text K/V with the
    freshly projected gen K/V and runs full attention. Queries are only the
    gen tokens. Nothing is written into the paged cache, and ``kv_cache_layer``
    is ignored.
    """

    def __init__(
        self,
        kv_cache: str,
        device: torch.device,
        dtype: torch.dtype,
        kv_config: KVConfig,
    ):
        self._kv_cache_name = kv_cache
        self._device = device
        self._dtype = dtype
        self._kv_config = kv_config
        self._plans: dict[str, _Plan] = {}
        # (request_id, label, layer_idx) -> frozen text K/V
        self._prefixes: dict[tuple[str, str, int], _Prefix] = {}
        logger.info(
            "AiterFmhaAttentionManager selected kv=%s dtype=%s; paged KV writes disabled",
            kv_cache,
            dtype,
        )

    def depends_on(self):
        return {self._kv_cache_name}

    @property
    def requires_kv_write(self) -> bool:
        return False

    def ingest_request(self, rid: str, overrides=None):
        del overrides
        self._drop_prefixes(rid)

    def remove_request(self, rid: str):
        self._drop_prefixes(rid)

    def reset_request(self, rid: str, free: bool = False):
        del free
        self._drop_prefixes(rid)

    def _drop_prefixes(self, rid: str) -> None:
        for key in [k for k in self._prefixes if k[0] == rid]:
            del self._prefixes[key]

    def plan(self, step: AttentionStep, ctx: StepContext):
        self.reset_default_cursors()
        if ctx.slot_lease is not None or ctx.is_preplan:
            raise RuntimeError(
                "aiter_fmha is eager-only: prefix concatenation allocates, "
                "so a captured graph cannot replay it. Set cuda_graph: false."
            )
        plan_outputs: KVPlanOutputs = ctx.plan_results.get(self._kv_cache_name)
        assert plan_outputs is not None, (
            f"aiter_fmha expected a KV plan from {self._kv_cache_name}"
        )
        self._plans = {
            plan_label: self._build_plan(kv_out, step.causal)
            for plan_label, kv_out in plan_outputs.items()
        }

    def _build_plan(self, kv_out: KVPlanOutput, causal: bool) -> _Plan:
        segments: list[_Segment] = []
        for view in kv_out.views:
            if view.start != 0:
                raise RuntimeError(
                    "aiter_fmha reads each stream from token 0; "
                    f"{view.label!r} of {view.request_id!r} starts at {view.start}"
                )
            prefix_len = view.length - view.to_compute
            if prefix_len < 0:
                raise RuntimeError(
                    f"KV plan to_compute {view.to_compute} exceeds length {view.length}"
                )
            segments.append(_Segment(
                request_id=view.request_id,
                label=view.label,
                prefix_len=prefix_len,
                q_len=view.to_compute,
                generation=view.generation,
            ))
        return _Plan(segments=tuple(segments), causal=causal)

    def _store_prefix(self, segment: _Segment, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self._prefixes[(segment.request_id, segment.label, layer_idx)] = _Prefix(
            generation=segment.generation,
            k=k.detach().clone(),
            v=v.detach().clone(),
        )

    def _load_prefix(self, segment: _Segment, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # KVManager.commit increments CacheStream.generation after the causal
        # prefill that stored this prefix, so a denoise view's generation is
        # not the generation captured at store time. The key is (request,
        # label, layer). A text-length mismatch still fails below.
        key = (segment.request_id, segment.label, layer_idx)
        entry = self._prefixes.get(key)
        if entry is None:
            raise RuntimeError(
                "aiter_fmha is missing the text-prefix K/V for "
                f"request {segment.request_id!r} label {segment.label!r} layer {layer_idx}. "
                "The understanding prefill must run on this same backend; "
                "paged FlashInfer writes are not read here."
            )
        if entry.k.shape[0] != segment.prefix_len:
            raise RuntimeError(
                f"stored prefix length {entry.k.shape[0]} != plan prefix {segment.prefix_len}"
            )
        return entry.k, entry.v

    @torch.compiler.disable
    def run(
        self,
        q: torch.Tensor,
        label: str | None = None,
        kv_cache_layer: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        # The page tensor is accepted because AttentionCallable always passes
        # it. This backend does not index it, copy it, or hand it to FlashInfer.
        del kv_cache_layer
        if label is None:
            label = self._default_label
        if layer_idx is None:
            layer_idx = self._default_layer_idx
        if k is None or v is None or layer_idx is None:
            raise RuntimeError("aiter_fmha requires k, v, and a layer index")
        plan = self._plans[label]
        if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape or q.shape[0] != k.shape[0]:
            raise RuntimeError(
                "packed Q/K/V must be [tokens, heads, dim] with one token axis; "
                f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
            )
        if q.shape[-1] != k.shape[-1]:
            raise RuntimeError(f"Q/K head dim mismatch {q.shape[-1]} vs {k.shape[-1]}")

        outs: list[torch.Tensor] = []
        offset = 0
        for segment in plan.segments:
            q_seg = q[offset:offset + segment.q_len]
            k_fresh = k[offset:offset + segment.q_len]
            v_fresh = v[offset:offset + segment.q_len]
            if k_fresh.shape[0] != segment.q_len:
                raise RuntimeError(
                    f"segment q_len {segment.q_len} ran past packed K of length {k.shape[0]}"
                )
            if segment.prefix_len:
                k_pref, v_pref = self._load_prefix(segment, layer_idx)
                k_seg = torch.cat([k_pref, k_fresh], dim=0)
                v_seg = torch.cat([v_pref, v_fresh], dim=0)
            else:
                k_seg, v_seg = k_fresh, v_fresh
            k_len = denoise_key_len(segment.prefix_len, segment.q_len)
            if k_seg.shape[0] != k_len:
                raise RuntimeError(f"key length {k_seg.shape[0]} != {k_len}")
            causal = step_causal(
                attention_causal=plan.causal, q_len=segment.q_len, k_len=k_len,
            )
            # Understanding prefill establishes the frozen prefix. A later
            # causal extension replaces it. Denoise is non-causal and must
            # leave the text prefix untouched: the next step recomputes gen K/V.
            if plan.causal:
                self._store_prefix(segment, layer_idx, k_seg, v_seg)

            head_dim = int(q_seg.shape[-1])
            q_b = _to_bshd(q_seg)
            k_b = _to_bshd(k_seg)
            v_b = _to_bshd(v_seg)
            out = aiter_flash_attn_func(q_b, k_b, v_b, causal=causal, head_dim=head_dim)
            if out.ndim != 4 or out.shape[0] != 1:
                raise RuntimeError(
                    "aiter.flash_attn_func must return BSHD [1, Sq, Hq, D], "
                    f"got {tuple(out.shape)}"
                )
            out = out[0]
            if out.shape != q_seg.shape:
                raise RuntimeError(
                    f"attention out {tuple(out.shape)} != query {tuple(q_seg.shape)}"
                )
            if out.dtype != q_seg.dtype:
                out = out.to(q_seg.dtype)
            outs.append(out)
            offset += segment.q_len
        if offset != q.shape[0]:
            raise RuntimeError(f"plan covered {offset} tokens, packed query has {q.shape[0]}")
        if not outs:
            return q[:0]
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
