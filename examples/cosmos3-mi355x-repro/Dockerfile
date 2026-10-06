# Cosmos3 Nano Policy DROID, 1x MI355X, on the official ROCm PyTorch image.
#
# Base, Docker Hub 2026-09-30:
#   rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0
#   sha256:cc9b00f90b85c97b015b040fa55c8d1b404b7cacc6ad57d74ee3451c97508da1
# Ubuntu 24.04, CPython 3.12, torch 2.12.0. ROCm 7.14 ships gfx950.
#
# vLLM-Omni v0.28.0 does not pin vllm. That release tracks the vLLM 0.28
# line. No ROCm 0.28 wheel is published
# (pypi.amd.com stops at 0.13; wheels.vllm.ai/rocm currently has 0.30.0+rocm723).
# v0.28.0 is built from source against this image's torch. Its
# requirements/build/rocm.txt pins torch==2.11.0 and is not installed.
#
# PYTORCH_ROCM_ARCH is exported only inside the vLLM compile RUN. Do not set
# it as an image ENV: FlashInfer fails to import when that variable is global.
#
# TorchCodec is not installed. The DROID action path uses
# torchvision.io.decode_image. Audio and video decoders are not on this path.
# Upstream load_image stays on decode_image. The image extension below is
# rebuilt with libpng because the 7.14.1 torchvision wheel ships PNG_FOUND off.
#
# M* is cloned from https://github.com/aamarnat/mstar branch mi355x-aiter-fmha.
# The build requires these implementation commits to be ancestors of HEAD:
#   7758d630852dfcdd5c8d9b19db1a58e63821bf8f
#          cosmos3: slice action output to raw action dim
#   98a848579f64aba8a5c60093970c5526abb514e9
#          attn: add direct AITER FMHA backend for ROCm
# They sit on cc8def9f11288d7f2f33ec9e65c39ad4e08ccb53. A later branch tip
# is accepted. The clone already contains that backend and the action-width
# slice. This image does not re-apply them.
#
# The request-wide profiler is not installed. Unprofiled latency does not use
# it. A chrome trace also needs a site .pth that clears
# ROCPROFILER_REGISTER_LIBRARY before torch is imported; see the README.

ARG BASE_IMAGE=rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0
FROM ${BASE_IMAGE}

ARG VLLM_REF=v0.28.0
ARG VLLM_OMNI_REF=v0.28.0
ARG VLLM_OMNI_VERSION=0.28.0+rocm
ARG FLASHINFER_REF=v0.6.18+amd.1
ARG MSTAR_REPO=https://github.com/aamarnat/mstar.git
ARG MSTAR_REF=mi355x-aiter-fmha
ARG MSTAR_COMMIT=7758d630852dfcdd5c8d9b19db1a58e63821bf8f
ARG MSTAR_ATTN_COMMIT=98a848579f64aba8a5c60093970c5526abb514e9
ARG AITER_VERSION=0.1.21+rocm7.2.manylinux.2.28
ARG AITER_WHEEL_URL=https://github.com/ROCm/aiter/releases/download/v0.1.21/amd_aiter-0.1.21+rocm7.2.manylinux.2.28-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git patch ca-certificates curl \
        cmake ninja-build g++ gcc make pkg-config patchelf \
        libpng-dev zlib1g-dev libnuma-dev python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    if ! command -v python >/dev/null 2>&1; then ln -sf "$(command -v python3)" /usr/local/bin/python; fi

# Identity of the base torch. Later steps refuse a different distribution.
RUN python3 - <<'PY'
import importlib.metadata as md
import json
import pathlib
import torch
ident = {
    "dist": md.version("torch"),
    "version": torch.__version__,
    "file": str(pathlib.Path(torch.__file__).resolve()),
    "hip": str(getattr(torch.version, "hip", None)),
}
pathlib.Path("/opt/cosmos3").mkdir(parents=True, exist_ok=True)
pathlib.Path("/opt/cosmos3/torch-identity.json").write_text(json.dumps(ident))
print("TORCH_BASE", json.dumps(ident))
pins = ["torch==" + ident["dist"]]
for name in ("torchvision", "torchaudio", "triton"):
    try:
        pins.append(name + "==" + md.version(name))
    except md.PackageNotFoundError:
        pass
pathlib.Path("/opt/cosmos3/constraints.txt").write_text("\n".join(pins) + "\n")
print("CONSTRAINTS", pins)
PY

# ROCm torchvision with libpng. The 7.14.1 wheel is built with PNG_FOUND off.
# Keep its _C ops and replace only the image extension, built from the matching
# torchvision tag with CPU image sources and libpng. A full _C rebuild links
# vision.cpp and vision_hip.cpp twice. A PyPI CUDA wheel is refused.
RUN python3 - <<'PY'
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

IDENT = Path("/opt/cosmos3/torch-identity.json")

def torch_identity():
    import torch
    return {
        "dist": md.version("torch"),
        "version": torch.__version__,
        "file": str(Path(torch.__file__).resolve()),
        "hip": str(getattr(torch.version, "hip", None)),
    }

def assert_torch_unchanged():
    import json
    before = json.loads(IDENT.read_text())
    after = torch_identity()
    if after != before:
        raise SystemExit(f"TORCH_CHANGED {before} -> {after}")

DECODE = r"""
import hashlib
import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torchvision.io import decode_image
rng = np.random.RandomState(0)
expected = rng.randint(0, 255, (480, 640, 3), dtype=np.uint8)
path = Path("/tmp/cosmos3-contract.png")
Image.fromarray(expected).save(path, format="PNG")
img = decode_image(str(path))
if img.dtype != torch.uint8 or tuple(img.shape) != (3, 480, 640):
    raise SystemExit(f"DECODE_LAYOUT {tuple(img.shape)} {img.dtype}")
got = img.permute(1, 2, 0).contiguous().numpy()
if not np.array_equal(got, expected):
    raise SystemExit("DECODE_PIXELS_MISMATCH")
digest = hashlib.sha256(img.numpy().tobytes()).hexdigest()
import torchvision
print("DECODE_SHAPE", tuple(img.shape))
print("DECODE_DTYPE", str(img.dtype).replace("torch.", ""))
print("DECODE_CHECKSUM", digest)
print("TORCHVISION", torchvision.__version__, torchvision.__file__)
"""

def decode_ok():
    env = os.environ.copy()
    env["TORCHVISION_WARN_WHEN_EXTENSION_LOADING_FAILS"] = "1"
    proc = subprocess.run([sys.executable, "-c", DECODE], text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        print("DECODE_STDOUT", proc.stdout[-2000:])
        print("DECODE_STDERR", proc.stderr[-4000:])
    else:
        print(proc.stdout)
    return proc.returncode == 0

def reject_cuda(name):
    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        return
    if "+cu" in version or "cuda" in version.lower():
        raise SystemExit(f"CUDA_WHEEL {name} {version}")

assert_torch_unchanged()
if decode_ok():
    reject_cuda("torchvision")
    print("TV_PNG_ALREADY_OK")
else:
    try:
        version = md.version("torchvision")
    except md.PackageNotFoundError:
        import torch
        # Release pairing: torch 2.12 -> torchvision 0.27.
        major, minor = torch.__version__.split("+", 1)[0].split(".")[:2]
        version = {"2.12": "0.27.0", "2.11": "0.26.0", "2.10": "0.25.0"}.get(
            f"{major}.{minor}", ""
        )
        if not version:
            raise SystemExit(f"TV_MISSING_AND_UNPAIRED torch={torch.__version__}")
        print("TV_MISSING_WILL_BUILD", version)
    else:
        print("TV_IMAGE_REBUILD", version)
    subprocess.check_call([
        "apt-get", "update",
    ])
    subprocess.check_call([
        "apt-get", "install", "-y", "--no-install-recommends", "libjpeg-dev",
    ])
    base = version.split("+", 1)[0]
    tag = "v" + base.split("a")[0].split("b")[0].split("rc")[0]
    src = Path("/tmp/cosmos3-vision")
    if src.exists():
        shutil.rmtree(src)
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--branch", tag,
        "https://github.com/pytorch/vision.git", str(src),
    ])
    setup_py = src / "setup.py"
    text = setup_py.read_text()
    marker = 'print("Building image extension")'
    start = text.find("    if IS_ROCM:\n", text.find(marker))
    end = text.find("    Extension = CppExtension\n", start)
    if start < 0 or end < 0:
        raise SystemExit("VISION_SETUP_UNEXPECTED image sources")
    text = text[:start] + text[end:]
    ext_old = "        make_C_extension(),\n        make_image_extension(),\n"
    ext_new = "        make_image_extension(),\n"
    if ext_old not in text:
        raise SystemExit("VISION_SETUP_UNEXPECTED extensions")
    text = text.replace(ext_old, ext_new, 1)
    libdir_old = 'library_dir = subprocess.run([libpng_config, "--libdir"], stdout=subprocess.PIPE).stdout.strip().decode("utf-8")'
    libdir_new = (
        'library_dir = subprocess.run([libpng_config, "--libdir"], stdout=subprocess.PIPE).stdout.strip().decode("utf-8")\n'
        '        if not os.path.isdir(library_dir):\n'
        '            library_dir = "/usr/lib/x86_64-linux-gnu"'
    )
    if libdir_old not in text:
        raise SystemExit("VISION_SETUP_UNEXPECTED libpng libdir")
    text = text.replace(libdir_old, libdir_new, 1)
    setup_py.write_text(text)
    image_cpp = src / "torchvision/csrc/io/image/image.cpp"
    cpp = image_cpp.read_text()
    cpp2, removed = re.subn(
        r'^.*image::(decode_jpegs_cuda|encode_jpegs_cuda).*\n',
        '',
        cpp,
        flags=re.M,
    )
    if removed != 2:
        raise SystemExit(f"VISION_IMAGE_CPP_UNEXPECTED removed={removed}")
    image_cpp.write_text(cpp2)
    env = os.environ.copy()
    env["FORCE_CUDA"] = "0"
    env["TORCHVISION_USE_NVJPEG"] = "0"
    env["TORCHVISION_USE_WEBP"] = "0"
    env["TORCHVISION_USE_PNG"] = "1"
    env["MAX_JOBS"] = "8"
    subprocess.check_call([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=src, env=env)
    import torchvision
    dest = Path(torchvision.__file__).resolve().parent
    built = list((src / "torchvision").glob("image*.so"))
    if len(built) != 1:
        raise SystemExit(f"TV_IMAGE_SO {built}")
    stock = sorted(dest.glob("image*.so"))
    for so in stock:
        print("STOCK_IMAGE", so.name, so.stat().st_size)
        print("STOCK_RPATH", subprocess.check_output(["patchelf", "--print-rpath", str(so)], text=True).strip())
        print("STOCK_NEEDED", subprocess.check_output(["patchelf", "--print-needed", str(so)], text=True).strip())
    for old in stock:
        old.unlink()
    shutil.copy2(built[0], dest / built[0].name)
    import torch
    installed = dest / built[0].name
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    # DT_RUNPATH does not apply to transitive NEEDED entries (libtorch_hip -> libamdhip64).
    # DT_RPATH does, and ldconfig covers the same ROCm SDK dirs for every later load.
    rpath_dirs = [str(torch_lib)]
    ldd = subprocess.check_output(["ldd", str(installed)], text=True)
    conf_dirs = []
    for line in ldd.splitlines():
        if "not found" not in line:
            continue
        name = line.strip().split()[0]
        hit = subprocess.run(
            ["bash", "-lc", f"find /opt/venv /opt/rocm /usr -name {name!r} 2>/dev/null | head -n 1"],
            text=True, capture_output=True,
        ).stdout.strip()
        print("TV_MISSING_LIB", name, hit or "ABSENT")
        if not hit:
            continue
        parent = str(Path(hit).parent)
        if parent not in conf_dirs:
            conf_dirs.append(parent)
        if parent not in rpath_dirs:
            rpath_dirs.append(parent)
    conf = Path("/etc/ld.so.conf.d/cosmos3-rocm-venv.conf")
    conf.write_text("".join(d + "\n" for d in conf_dirs))
    subprocess.check_call(["ldconfig"])
    subprocess.check_call([
        "patchelf", "--force-rpath", "--set-rpath", ":".join(rpath_dirs), str(installed),
    ])
    ldd = subprocess.check_output(["ldd", str(installed)], text=True)
    print("TV_IMAGE_SO_INSTALLED", installed, installed.stat().st_size)
    print("TV_IMAGE_RPATH", ":".join(rpath_dirs))
    print(ldd)
    dlopen = subprocess.run(
        [sys.executable, "-c",
         "import ctypes, torch\n"
         f"ctypes.CDLL({str(installed)!r}, mode=ctypes.RTLD_GLOBAL)\n"
         "print('DLOPEN_OK')\n"],
        text=True, capture_output=True,
    )
    print("DLOPEN_RC", dlopen.returncode)
    print(dlopen.stdout)
    print(dlopen.stderr[-2000:])
    if dlopen.returncode != 0:
        raise SystemExit("TV_IMAGE_DLOPEN_FAIL")
    shutil.rmtree(src)
    assert_torch_unchanged()
    reject_cuda("torchvision")
    if not decode_ok():
        raise SystemExit("TV_SOURCE_DECODE_FAIL")
    print("TV_PNG_SOURCE_OK", md.version("torchvision"))

pins = ["torch==" + md.version("torch")]
for name in ("torchvision", "torchaudio", "triton"):
    try:
        pins.append(name + "==" + md.version(name))
    except md.PackageNotFoundError:
        pass
Path("/opt/cosmos3/constraints.txt").write_text("\n".join(pins) + "\n")
print("CONSTRAINTS", pins)
PY

RUN python3 -m pip install --no-cache-dir \
        "packaging>=24.2" \
        "setuptools>=77.0.3,<80" \
        "setuptools-scm>=8" \
        "setuptools-rust>=1.9" \
        wheel \
        "jinja2>=3.1.6" \
        ninja \
        regex && \
    git clone --depth 1 --branch "${VLLM_REF}" \
        https://github.com/vllm-project/vllm.git /opt/vllm && \
    cd /opt/vllm && \
    python3 -m pip install --no-cache-dir --no-deps \
        --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
        "rocm-sdk-devel==7.14.1" && \
    rocm-sdk init && \
    export ROCM_PATH="$(rocm-sdk path --root)" && \
    export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" && \
    if [ -d "${ROCM_PATH}/lib/llvm/amdgcn/bitcode" ]; then \
        export HIP_DEVICE_LIB_PATH="${ROCM_PATH}/lib/llvm/amdgcn/bitcode"; \
    fi && \
    echo "ROCM_PATH=${ROCM_PATH}" && \
    test -f "${ROCM_PATH}/lib/cmake/hip-lang/hip-lang-config.cmake" && \
    export PYTORCH_ROCM_ARCH=gfx950 && \
    export GPU_ARCHS=gfx950 && \
    export VLLM_TARGET_DEVICE=rocm && \
    export MAX_JOBS=32 && \
    export CMAKE_BUILD_PARALLEL_LEVEL=32 && \
    export VLLM_USE_PRECOMPILED=0 && \
    python3 -m pip wheel . --no-build-isolation --no-deps -w /tmp/vllm-wheels && \
    python3 -m pip install --no-cache-dir --no-deps /tmp/vllm-wheels/vllm-*.whl && \
    rm -rf /tmp/vllm-wheels /opt/vllm/build && \
    python3 - <<'PY'
import importlib.metadata as md
import json
import pathlib
import torch
before = json.loads(pathlib.Path("/opt/cosmos3/torch-identity.json").read_text())
after = {
    "dist": md.version("torch"),
    "version": torch.__version__,
    "file": str(pathlib.Path(torch.__file__).resolve()),
    "hip": str(getattr(torch.version, "hip", None)),
}
if after != before:
    raise SystemExit(f"TORCH_CHANGED {before} -> {after}")
version = md.version("vllm")
if not version.startswith("0.28.0"):
    raise SystemExit(f"VLLM_VERSION {version}")
print("VLLM_BUILD_OK", version, "torch", after["version"])
PY

# Runtime dependencies from the v0.28 ROCm requirements. The constraint file
# keeps the base torch, torchvision, and triton.
RUN python3 -m pip install --no-cache-dir \
        -c /opt/cosmos3/constraints.txt \
        -r /opt/vllm/requirements/rocm.txt && \
    python3 - <<'PY'
import importlib.metadata as md
import json
import pathlib
import torch
before = json.loads(pathlib.Path("/opt/cosmos3/torch-identity.json").read_text())
after = {
    "dist": md.version("torch"),
    "version": torch.__version__,
    "file": str(pathlib.Path(torch.__file__).resolve()),
    "hip": str(getattr(torch.version, "hip", None)),
}
if after != before:
    raise SystemExit(f"TORCH_CHANGED {before} -> {after}")
for name in ("torchvision", "torchaudio"):
    try:
        version = md.version(name)
    except md.PackageNotFoundError:
        continue
    if "+cu" in version or "cuda" in version.lower():
        raise SystemExit(f"CUDA_WHEEL {name} {version}")
print("VLLM_REQUIREMENTS_OK", md.version("vllm"))
PY

RUN git clone --depth 1 --branch "${VLLM_OMNI_REF}" \
        https://github.com/vllm-project/vllm-omni.git /opt/vllm-omni && \
    cd /opt/vllm-omni && \
    VLLM_OMNI_TARGET_DEVICE=rocm \
    VLLM_OMNI_VERSION_OVERRIDE="${VLLM_OMNI_VERSION}" \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps . && \
    python3 -m pip install --no-cache-dir \
        -c /opt/cosmos3/constraints.txt \
        -r /opt/vllm-omni/requirements/rocm.txt && \
    python3 - <<'PY'
import importlib.metadata as md
import json
import pathlib
import torch
before = json.loads(pathlib.Path("/opt/cosmos3/torch-identity.json").read_text())
after = {
    "dist": md.version("torch"),
    "version": torch.__version__,
    "file": str(pathlib.Path(torch.__file__).resolve()),
    "hip": str(getattr(torch.version, "hip", None)),
}
if after != before:
    raise SystemExit(f"TORCH_CHANGED {before} -> {after}")
vllm = md.version("vllm")
if not vllm.startswith("0.28.0"):
    raise SystemExit(f"VLLM_CHANGED {vllm}")
omni = md.version("vllm-omni")
if not omni.startswith("0.28.0"):
    raise SystemExit(f"VLLM_OMNI_VERSION {omni}")
print("VLLM_OMNI_OK", omni, "vllm", vllm)
PY

# Installed vLLM-Omni v0.28.0 decodes video after the joint denoise. The
# benchmark returns float32 actions only when skip_video_decode is set.
RUN python3 - <<'PY'
import pathlib
path = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py"
)
if not path.is_file():
    raise SystemExit(f"OMNI_PIPELINE_MISSING {path}")
text = path.read_text()
marker = "Skipping VAE decode; returning actions only"
if marker in text:
    print(f"OMNI_ACTIONS_ONLY_ALREADY {path}")
else:
    anchor = (
        "        # --- Decode ---\n"
        "        if _is_rank_zero():\n"
        "            logger.info(\"Decoding video...\")\n"
    )
    insert = (
        "        skip_video_decode = bool(self._get_sp_param(sp, \"skip_video_decode\", False))\n"
        "        if skip_video_decode and action_enabled and not sound_enabled:\n"
        "            if action_latents is None or raw_action_dim is None or domain_id is None:\n"
        "                raise ValueError(\"Cosmos3 action generation finished without action latents.\")\n"
        "            if _is_rank_zero():\n"
        "                logger.info(\"Skipping VAE decode; returning actions only\")\n"
        "            action = action_latents[:, :, :raw_action_dim].detach().to(dtype=torch.float32).cpu()\n"
        "            return DiffusionOutput(\n"
        "                output={\n"
        "                    \"payload\": {\"actions\": action},\n"
        "                    \"metadata\": {\n"
        "                        \"actions\": {\n"
        "                            \"raw_action_dim\": int(raw_action_dim),\n"
        "                            \"action_mode\": action_mode,\n"
        "                            \"domain_id\": domain_id,\n"
        "                        },\n"
        "                        \"common\": {\n"
        "                            \"action_only_output\": True,\n"
        "                            \"skip_video_decode\": True,\n"
        "                        },\n"
        "                    },\n"
        "                },\n"
        "            )\n"
        "\n"
    )
    if text.count(anchor) != 1:
        raise SystemExit(f"OMNI_DECODE_ANCHOR {text.count(anchor)}")
    path.write_text(text.replace(anchor, insert + anchor, 1))
    print(f"OMNI_ACTIONS_ONLY_OK {path}")
PY


# Pure-Python FlashInfer wheel. Build deps are restored afterward so the
# image setuptools pin is not left at the FlashInfer build requirement.
# Import is deferred when no GPU is attached to the build.
RUN python3 - <<'PY'
import importlib.metadata as md, pathlib
pathlib.Path("/tmp/setuptools-before.txt").write_text(md.version("setuptools"))
print("SETUPTOOLS_BEFORE", md.version("setuptools"))
PY
RUN python3 -m pip install --no-cache-dir "setuptools>=80" "setuptools-scm>=9.2" "packaging>=24" && \
    git clone --depth 1 --branch "${FLASHINFER_REF}" \
        https://github.com/AMD-Ecosystem/flashinfer.git /tmp/flashinfer && \
    SETUPTOOLS_SCM_PRETEND_VERSION="${FLASHINFER_REF}" \
      python3 -m pip wheel /tmp/flashinfer --no-build-isolation --no-deps -w /tmp/fi-wheels && \
    python3 -m pip install --no-cache-dir --no-deps /tmp/fi-wheels/*.whl && \
    python3 -m pip install --no-cache-dir --no-deps filelock jinja2 ninja && \
    SETUPTOOLS_BEFORE="$(cat /tmp/setuptools-before.txt)" && \
    python3 -m pip install --no-cache-dir --no-deps "setuptools==${SETUPTOOLS_BEFORE}" && \
    rm -rf /tmp/flashinfer /tmp/fi-wheels && \
    python3 - <<'PY'
import torch
print("BUILD_CUDA", torch.cuda.is_available())
try:
    import flashinfer
except RuntimeError as exc:
    text = str(exc)
    if "CUDA GPUs" not in text and "HIP" not in text:
        raise
    print("FLASHINFER_IMPORT_DEFERRED", text)
else:
    print("FLASHINFER_IMPORT_OK", flashinfer.__file__)
PY

# Full history so a later branch tip still contains both implementation commits.
RUN git clone --branch "${MSTAR_REF}" "${MSTAR_REPO}" /opt/mstar && \
    cd /opt/mstar && \
    git merge-base --is-ancestor "${MSTAR_ATTN_COMMIT}" HEAD && \
    git merge-base --is-ancestor "${MSTAR_COMMIT}" HEAD && \
    python3 -m pip install --no-cache-dir --no-deps . && \
    python3 -m pip install --no-cache-dir \
        -c /opt/cosmos3/constraints.txt \
        fastapi PyYAML pyzmq msgpack python-multipart requests uvicorn gdown \
        datasets av "diffusers==0.40.0" huggingface-hub Pillow safetensors \
        "transformers>=5.10.1,<5.15" einops regex && \
    python3 - <<'PY'
import importlib.metadata as md
import json
import pathlib
import torch
before = json.loads(pathlib.Path("/opt/cosmos3/torch-identity.json").read_text())
after = {
    "dist": md.version("torch"),
    "version": torch.__version__,
    "file": str(pathlib.Path(torch.__file__).resolve()),
    "hip": str(getattr(torch.version, "hip", None)),
}
if after != before:
    raise SystemExit(f"TORCH_CHANGED {before} -> {after}")
print("MSTAR_OK", md.version("mstar-ai"))
PY

# amd-aiter 0.1.21 is the ROCm 7.2 cp312 wheel. It links
# libtorch and does not declare a torch dependency; --no-deps keeps this torch.
# flydsl==0.3.2 is the wheel's pinned runtime dependency.
RUN export AITER_VERSION="${AITER_VERSION}" AITER_WHEEL_URL="${AITER_WHEEL_URL}" && \
    python3 - <<'PY'
import importlib.metadata as md
import json
import os
import pathlib
import subprocess
import sys
import urllib.request

expected = os.environ["AITER_VERSION"]
wheel_url = os.environ["AITER_WHEEL_URL"]
before = {
    "torch": md.version("torch"),
    "vllm": md.version("vllm"),
    "vllm-omni": md.version("vllm-omni"),
    "torchvision": md.version("torchvision"),
}
print("AITER_UPGRADE_BEFORE", json.dumps(before))
for name in ("amd-aiter", "aiter"):
    try:
        md.distribution(name)
    except md.PackageNotFoundError:
        continue
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", name])
wheel_path = pathlib.Path("/tmp") / wheel_url.rsplit("/", 1)[-1]
urllib.request.urlretrieve(wheel_url, wheel_path)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
    "--force-reinstall", "flydsl==0.3.2", str(wheel_path),
])
wheel_path.unlink()
installed = md.version("amd-aiter")
if installed != expected:
    raise SystemExit(f"AITER_VERSION {installed!r} != {expected!r}")
after = {
    "torch": md.version("torch"),
    "vllm": md.version("vllm"),
    "vllm-omni": md.version("vllm-omni"),
    "torchvision": md.version("torchvision"),
}
for key, value in before.items():
    if after[key] != value:
        raise SystemExit(f"{key} changed {value!r} -> {after[key]!r}")
inits = []
for entry in map(pathlib.Path, sys.path):
    init = entry / "aiter" / "__init__.py"
    if init.is_file():
        inits.append(init.resolve())
if len(set(inits)) != 1:
    raise SystemExit(f"AITER_COPY_COUNT {inits}")
try:
    import aiter
except RuntimeError as exc:
    # docker build has no /dev/kfd, and this wheel's import calls rocminfo.
    if "rocminfo" not in str(exc):
        raise
    print("AITER_IMPORT_DEFERRED", installed, "flydsl", md.version("flydsl"), "init", inits[0])
else:
    print("AITER_IMPORT_OK", installed, "flydsl", md.version("flydsl"), "init", inits[0])
PY

# vLLM 0.28 selects RocmPlatform only when `import amdsmi` works. This base
# ships that package under the ROCm SDK tree, not on sys.path.
# FlashInfer's ROCm JIT calls /opt/rocm/bin/amdclang++ and includes ck_tile.
# This base keeps both in the SDK / aiter wheel, not under /opt/rocm.
RUN ln -sfn /opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel /opt/rocm && \
    ln -sfn /opt/venv/lib/python3.12/site-packages/aiter_meta/3rdparty/composable_kernel/include/ck_tile /usr/include/ck_tile && \
    test -x /opt/rocm/bin/amdclang++ && \
    test -f /usr/include/ck_tile/host/stream_config.hpp && \
    python3 -m pip install --no-cache-dir \
        -c /opt/cosmos3/constraints.txt \
        pybind11 && \
    python3 - <<'PY'
from pathlib import Path
site = Path("/opt/venv/lib/python3.12/site-packages")
amd_smi = site / "_rocm_sdk_core/share/amd_smi"
if not (amd_smi / "amdsmi/amdsmi_interface.py").is_file():
    raise SystemExit(f"AMDSMI_MISSING {amd_smi}")
(site / "rocm-amd-smi.pth").write_text(str(amd_smi) + "\n")
import sys
sys.path.insert(0, str(amd_smi))
import amdsmi
import importlib.metadata as md
print("AMDSMI_IMPORT_OK", amdsmi.__file__)
print("PYBIND11", md.version("pybind11"))
PY

WORKDIR /opt/mstar
CMD ["python3", "-c", "import torch,vllm,torchvision; print(torch.__version__, torch.version.hip, vllm.__version__, torchvision.__version__)"]
