#!/usr/bin/env python3
"""vLLM-Omni client for the 1-GPU Cosmos3 run. Actions only, float32 [1, 16, 8].

The joint video-latent + action denoise still runs. ``modalities`` stays
``["video"]`` because that is how this pipeline selects the video-latent
policy path. The only multimodal input is the image. ``skip_video_decode``
drops the post-denoise VAE decode.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from cosmos3_contract import (
    CHUNK,
    FRAMES,
    HEIGHT,
    KNOBS,
    OUTPUT_DTYPE,
    OUTPUT_SHAPE,
    PROMPT,
    RAW_DIM,
    WIDTH,
    assert_png,
    load_manifest,
)


def build_request(image: Image.Image):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    params = OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        num_frames=FRAMES,
        num_inference_steps=KNOBS["num_inference_steps"],
        guidance_scale=1.0,
        seed=KNOBS["seed"],
        num_outputs_per_prompt=1,
        extra_args={
            "action_mode": "policy",
            "domain_name": "droid_lerobot",
            "action_chunk_size": CHUNK,
            "raw_action_dim": RAW_DIM,
            "flow_shift": 5.0,
            "skip_video_decode": True,
        },
    )
    prompt = {
        "prompt": PROMPT,
        "modalities": ["video"],
        "multi_modal_data": {"image": image},
    }
    return prompt, params


def check_output(out) -> dict:
    images = getattr(out, "images", None) or []
    if images:
        raise RuntimeError(f"video/image output present: {type(images).__name__} len={len(images)}")
    mm = getattr(out, "multimodal_output", None) or {}
    if not isinstance(mm, dict):
        raise RuntimeError(f"multimodal_output type {type(mm)}")
    if "video" in mm:
        raise RuntimeError("multimodal_output contains video")
    meta = mm.get("metadata") if isinstance(mm.get("metadata"), dict) else {}
    common = meta.get("common") if isinstance(meta.get("common"), dict) else {}
    if common.get("skip_video_decode") is not True:
        raise RuntimeError(f"skip_video_decode was not set on the response metadata: {meta}")
    actions = mm.get("actions")
    if actions is None:
        raise RuntimeError(f"actions missing; keys={sorted(mm)}")
    import torch

    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)
    if tuple(actions.shape) != tuple(OUTPUT_SHAPE):
        raise RuntimeError(f"actions shape {tuple(actions.shape)} != {OUTPUT_SHAPE}")
    if actions.dtype != torch.float32:
        raise RuntimeError(f"actions dtype {actions.dtype} != float32")
    if not torch.isfinite(actions).all():
        raise RuntimeError("action values are not finite")
    return {
        "output_modalities": ["action"],
        "shape": [int(dim) for dim in actions.shape],
        "dtype": OUTPUT_DTYPE,
        "finite": True,
        "skip_video_decode": True,
        "multimodal_keys": sorted(k for k in mm if k != "metadata"),
    }


def run_one(omni, prompt, params) -> dict:
    t0 = time.perf_counter()
    outputs = omni.generate(prompt, params, use_tqdm=False)
    e2e_ms = (time.perf_counter() - t0) * 1000.0
    out = outputs[0] if isinstance(outputs, list) else outputs
    rec = check_output(out)
    rec["e2e_ms"] = e2e_ms
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-sha", required=True)
    parser.add_argument("--model", default=KNOBS["model"])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    digest = assert_png(args.image, args.image_sha)
    image = Image.open(args.image).convert("RGB")
    manifest = load_manifest()
    from vllm_omni.entrypoints.omni import Omni

    t0 = time.perf_counter()
    omni = Omni(
        model=args.model,
        model_config={"guardrails": False},
        enforce_eager=False,
    )
    engine_init_s = time.perf_counter() - t0
    print(f"OMNI_ENGINE_INIT_S {engine_init_s:.2f}", flush=True)
    prompt, params = build_request(image)
    if "video" in (prompt.get("multi_modal_data") or {}):
        raise SystemExit("video input is not part of this contract")

    warmups = []
    for i in range(args.warmups):
        rec = run_one(omni, prompt, params)
        warmups.append(rec)
        print(f"OMNI_WARMUP {i} e2e_ms={rec['e2e_ms']:.2f}", flush=True)
    records = []
    for i in range(args.repeats):
        rec = run_one(omni, prompt, params)
        records.append(rec)
        print(f"OMNI_STEADY {i} e2e_ms={rec['e2e_ms']:.2f}", flush=True)
    lat = [r["e2e_ms"] for r in records]
    mean_ms = statistics.fmean(lat)
    metrics = {
        "n_steady": len(records),
        "e2e_ms": {
            "mean": mean_ms,
            "p50": statistics.median(lat),
            "min": min(lat),
            "max": max(lat),
            "all": lat,
        },
        "action_rate_hz": CHUNK / (mean_ms / 1000.0),
        "profiled": False,
    }
    payload = {
        "stack": "vllm-omni",
        "phase": "unprofiled",
        "engine_init_s": engine_init_s,
        "knobs": KNOBS,
        "path": "policy_joint_denoise_skip_video_decode",
        "prompt_modalities_field": ["video"],
        "prompt_modalities_note": (
            "Selects the joint video-latent policy denoise. The multimodal "
            "input is the image only; skip_video_decode drops the VAE decode."
        ),
        "png_sha256": digest,
        "version_manifest": manifest,
        "warmup_discarded": warmups,
        "records": records,
        "metrics": metrics,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print("OMNI_METRICS " + json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
