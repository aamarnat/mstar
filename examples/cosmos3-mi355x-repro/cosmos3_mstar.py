#!/usr/bin/env python3
"""M* client for the 1-GPU Cosmos3 run. Actions only, float32 [1, 16, 8]."""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from cosmos3_contract import (
    CHUNK,
    KNOBS,
    OUTPUT_DTYPE,
    OUTPUT_SHAPE,
    PROMPT,
    assert_png,
    load_manifest,
)


def encode_multipart(fields: dict[str, str], file_bytes: bytes) -> tuple[bytes, str]:
    boundary = "----cosmos3-boundary"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.append(value.encode())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        b'Content-Disposition: form-data; name="files"; filename="obs.png"\r\n'
        b"Content-Type: image/png\r\n\r\n"
    )
    chunks.append(file_bytes)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def one_request(url: str, image: bytes, timeout: float, request_id: str) -> dict:
    body, boundary = encode_multipart(
        {
            "text": PROMPT,
            "input_modalities": "image,text",
            "output_modalities": "action",
            "streaming": "false",
            "model_kwargs": json.dumps(
                {
                    "action_mode": "policy",
                    "domain_name": "droid_lerobot",
                    "action_chunk_size": CHUNK,
                    "num_frames": KNOBS["num_frames"],
                    "raw_action_dim": KNOBS["raw_action_dim"],
                    "num_inference_steps": KNOBS["num_inference_steps"],
                    "guidance_scale": 1.0,
                    "flow_shift": 5.0,
                    "size": f"{KNOBS['width']}x{KNOBS['height']}",
                    "seed": KNOBS["seed"],
                }
            ),
            "request_id": request_id,
        },
        image,
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
            status = resp.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:2000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    e2e_ms = (time.perf_counter() - t0) * 1000.0
    outputs = payload.get("outputs") or {}
    if sorted(outputs) != ["action"]:
        raise RuntimeError(f"output modalities {sorted(outputs)} != ['action']")
    chunks = outputs["action"]
    if len(chunks) != 1:
        raise RuntimeError(f"expected one action chunk, got {len(chunks)}")
    meta = chunks[0].get("metadata") or {}
    raw = base64.b64decode(chunks[0].get("data") or "")
    expect = int(np.prod(OUTPUT_SHAPE)) * 4
    if len(raw) != expect:
        raise RuntimeError(f"action bytes {len(raw)} != {expect} for shape {OUTPUT_SHAPE}")
    arr = np.frombuffer(raw, dtype=np.float32).reshape(OUTPUT_SHAPE)
    if str(arr.dtype) != OUTPUT_DTYPE:
        raise RuntimeError(f"dtype {arr.dtype}")
    if list(meta.get("shape") or []) != OUTPUT_SHAPE:
        raise RuntimeError(f"metadata shape {meta.get('shape')} != {OUTPUT_SHAPE}")
    if meta.get("dtype") != OUTPUT_DTYPE:
        raise RuntimeError(f"metadata dtype {meta.get('dtype')}")
    if not np.isfinite(arr).all():
        raise RuntimeError("action values are not finite")
    return {
        "e2e_ms": e2e_ms,
        "http_status": status,
        "output_modalities": ["action"],
        "shape": OUTPUT_SHAPE,
        "dtype": OUTPUT_DTYPE,
        "finite": True,
        "request_id": request_id,
    }


def summarize(records: list[dict]) -> dict:
    lat = [r["e2e_ms"] for r in records]
    mean_ms = statistics.fmean(lat)
    return {
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080/generate")
    parser.add_argument("--out", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-sha", required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    digest = assert_png(args.image, args.image_sha)
    image = Path(args.image).read_bytes()
    manifest = load_manifest()
    warmups = []
    for i in range(args.warmups):
        rec = one_request(args.url, image, args.timeout, f"warmup-{i}")
        warmups.append(rec)
        print(f"MSTAR_WARMUP {i} e2e_ms={rec['e2e_ms']:.2f}", flush=True)
    records = []
    for i in range(args.repeats):
        rec = one_request(args.url, image, args.timeout, f"steady-{i}")
        records.append(rec)
        print(f"MSTAR_STEADY {i} e2e_ms={rec['e2e_ms']:.2f}", flush=True)
    metrics = summarize(records)
    payload = {
        "stack": "mstar",
        "phase": "unprofiled",
        "knobs": KNOBS,
        "png_sha256": digest,
        "version_manifest": manifest,
        "warmup_discarded": warmups,
        "records": records,
        "metrics": metrics,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print("MSTAR_METRICS " + json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
