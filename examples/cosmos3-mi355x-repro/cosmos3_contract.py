#!/usr/bin/env python3
"""Shared request contract for the 1-GPU Cosmos3 Nano Policy DROID run.

The observation PNG is uint8 RGB [480, 640, 3] from ``np.random.RandomState(0)``.
Action rate is 16 divided by mean steady seconds. There is no frame rate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

PROMPT = "Pick up the banana and place it in the bowl."
MODEL = "nvidia/Cosmos3-Nano-Policy-DROID"
HEIGHT = 480
WIDTH = 640
STEPS = 30
FRAMES = 17
CHUNK = 16
RAW_DIM = 8
SEED = 0
OUTPUT_SHAPE = [1, CHUNK, RAW_DIM]
OUTPUT_DTYPE = "float32"

KNOBS = {
    "model": MODEL,
    "prompt": PROMPT,
    "action_mode": "policy",
    "domain_name": "droid_lerobot",
    "action_chunk_size": CHUNK,
    "num_frames": FRAMES,
    "raw_action_dim": RAW_DIM,
    "num_inference_steps": STEPS,
    "guidance_scale": 1.0,
    "flow_shift": 5.0,
    "height": HEIGHT,
    "width": WIDTH,
    "seed": SEED,
    "compiled": True,
    "batch": 1,
    "input_modalities": ["image", "text"],
    "output_modalities": ["action"],
    "output_shape": OUTPUT_SHAPE,
    "output_dtype": OUTPUT_DTYPE,
}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _file(module: str) -> str | None:
    try:
        mod = __import__(module)
    except Exception as exc:  # noqa: BLE001 — the manifest reports the import error
        return f"import-error: {type(exc).__name__}: {exc}"
    return getattr(mod, "__file__", None)


def load_manifest() -> dict:
    """Package versions of the image that served the request."""
    import torch

    gcn = []
    try:
        if torch.cuda.is_available():
            gcn = [
                torch.cuda.get_device_properties(i).gcnArchName
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # noqa: BLE001
        gcn = [f"error: {exc}"]
    manifest = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "torch_file": torch.__file__,
        "vllm": _version("vllm"),
        "vllm_file": _file("vllm"),
        "vllm_omni": _version("vllm-omni"),
        "vllm_omni_file": _file("vllm_omni"),
        "mstar": _version("mstar-ai"),
        "mstar_file": _file("mstar"),
        "flashinfer": _version("amd-flashinfer") or _version("flashinfer"),
        "aiter": _version("amd-aiter") or _version("aiter"),
        "gcn_arch": gcn,
    }
    print("VERSION_MANIFEST " + json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


def assert_png(path: str | Path, expected_sha: str) -> str:
    """Reject a PNG whose file hash or pixels are not the RandomState(0) image."""
    import numpy as np
    from PIL import Image

    digest = sha256_file(path)
    if digest != expected_sha:
        raise SystemExit(f"PNG sha256 {digest} != expected {expected_sha}")
    image = np.asarray(Image.open(path))
    if image.dtype != np.uint8 or tuple(image.shape) != (HEIGHT, WIDTH, 3):
        raise SystemExit(f"PNG pixels {image.dtype} {image.shape} != uint8 {(HEIGHT, WIDTH, 3)}")
    rng = np.random.RandomState(0)
    expected = rng.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    if not np.array_equal(image, expected):
        raise SystemExit("PNG pixels do not match RandomState(0)")
    return digest


def write_png(path: str | Path) -> str:
    """Write the observation PNG and return its sha256.

    Pillow's compressed bytes depend on the Pillow build. Generate this file
    inside the image, then pass the printed digest to the clients.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(0)
    arr = rng.randint(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    dest = Path(path)
    Image.fromarray(arr).save(dest, format="PNG")
    digest = sha256_file(dest)
    assert_png(dest, digest)
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(write_png(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
