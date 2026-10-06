# Cosmos3 Nano Policy DROID, 1× MI355X

M* (`attention_backend: aiter_fmha`) against vLLM-Omni on one AMD MI355X. Both serve `nvidia/Cosmos3-Nano-Policy-DROID` in policy mode. The action output is float32 `[1, 16, 8]`. Action rate is `16 / e2e_seconds`. There is no frame rate.

The image is official `rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0` (torch 2.12.0). It rebuilds torchvision's `image.so` with libpng so `torchvision.io.decode_image` works, and installs vLLM 0.28.0, vLLM-Omni 0.28.0, and amd-aiter 0.1.21. M* is cloned from `https://github.com/aamarnat/mstar` branch `mi355x-aiter-fmha`. The Dockerfile requires these commits to be ancestors of that branch tip, and does not re-apply them:

- `9e9f3ac9085f53b7ea7b533bfa95dbdfd2b7e21c` slices the action output to `raw_action_dim`
- `f0223c0b39580a212a9c8f03d9295126bf1752b2` adds the direct AITER FMHA backend

Those commits sit on `166ddb089ebdc8044a66a72d7ff401ae53398c2e`. `PYTORCH_ROCM_ARCH` is set only inside the vLLM compile step. It is not an image environment variable.

The branch yaml `configs/cosmos3_droid.yaml` is a different serving setup. This run uses `examples/cosmos3-mi355x-repro/cosmos3-droid-aiter-fmha.yaml`.

## Bundle contents

- `cosmos3_contract.py`
  - This is the shared request for both clients. It writes the observation PNG and holds the request constants.
  - Run it with `--out` set to the PNG path. It prints the SHA-256 of the file it writes. Generate that PNG inside the image, because the compressed bytes depend on that image's Pillow build.
  - The request is the prompt `Pick up the banana and place it in the bowl.`, a 480×640 image, seed 0, 16 actions, 30 denoise steps, and float32 output `[1, 16, 8]`. The PNG is uint8 RGB from `np.random.RandomState(0)`. `assert_png` checks the SHA-256 and those pixels. `load_manifest` prints the package versions of the image that served the request, and both clients call it.

- `cosmos3_mstar.py`
  - This is the HTTP client for the M* `/generate` endpoint. It uses that contract.
  - Each call is a non-streaming multipart POST carrying the prompt, the PNG, and action-only output. Run it with `--url`, `--image`, `--image-sha`, and `--out`.
  - It checks the PNG with `assert_png` before any request. A response must be JSON with one base64 action, finite values, metadata that matches the contract, and output keys exactly `action`. The `--out` file records the version manifest, the discarded warmups, the per-request timings, and summary metrics (`mean`, `p50`, `min`, `max`, `all`, and `action_rate_hz`).

- `cosmos3_omni.py`
  - This is the in-process vLLM-Omni client. It uses that contract.
  - It constructs `vllm_omni.entrypoints.omni.Omni` with `model_config={"guardrails": False}` and `enforce_eager=False`, then calls `Omni.generate`. The prompt `modalities` field is `["video"]`, which selects the joint video-latent policy path. The only multimodal input is the image, and `skip_video_decode` drops the VAE decode. Run it with `--image`, `--image-sha`, and `--out`.
  - Each result must be finite actions that match the contract, with `skip_video_decode` set, an empty image list, and no `video` key. The `--out` file records the same timing metrics plus `engine_init_s`.

- `cosmos3-droid-aiter-fmha.yaml`
  - This is the one-GPU M* serve config. It uses that contract.
  - It sets `model: cosmos3_droid`, `attention_backend: aiter_fmha`, `compile_denoise: true`, `cuda_graph: false`, and `enable_sound: false`. `max_seq_len` is 8192. The DiT, VAE encoder, and VAE decoder node groups are all rank 0.
  - `aiter_fmha` does not write paged K/V. The KV pool is `max_num_pages: 64`.

- `Dockerfile`
  - This builds the M* and vLLM-Omni stacks from the official ROCm PyTorch image named above.
  - The M* clone is `https://github.com/aamarnat/mstar` branch `mi355x-aiter-fmha`.
  - The build checks that `f0223c0b39580a212a9c8f03d9295126bf1752b2` and `9e9f3ac9085f53b7ea7b533bfa95dbdfd2b7e21c` are ancestors of `HEAD`.

- `run_bench.sh`
  - This is the shell script for the unprofiled benchmark.
  - It generates the observation PNG, starts M* from the yaml, checks the server log, times five M* requests, then times five vLLM-Omni requests.
  - It writes `obs.png`, `mstar-unprofiled.json`, and `omni-unprofiled.json` in the current directory.

## Token and cache

Put the Hugging Face token in the environment before `docker run`. Pass it through with `-e HF_TOKEN` (the name only). Do not put the token on the command line, in a file, or in the image.

```bash
export HF_TOKEN=...
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
```

`HF_HOME` defaults to `$HOME/hf-cache` in the example above. Set it to the directory that should hold the download. The first M* and vLLM-Omni runs download `nvidia/Cosmos3-Nano-Policy-DROID` into that cache.

## Build

From a checkout of `https://github.com/aamarnat/mstar` branch `mi355x-aiter-fmha`, change into this directory and build. The machine must be able to pull the base image.

```bash
cd examples/cosmos3-mi355x-repro
DOCKER_BUILDKIT=1 docker build -f Dockerfile -t cosmos3-mi355x:repro .
```

## Request

Policy mode, domain `droid_lerobot`, prompt `Pick up the banana and place it in the bowl.`, one PNG of `np.random.RandomState(0)` uint8 `[480, 640, 3]`, 16-action chunk, `num_frames` 17, 30 denoise steps, guidance 1.0, flow shift 5.0, `raw_action_dim` 8, seed 0, compiled, batch 1. Input is image and text. There is no video input. M* output modalities are action only. vLLM-Omni still sets prompt `modalities` to `["video"]` so the pipeline takes the joint video-latent policy path; `skip_video_decode` drops the VAE decode and the response is actions only.

Discard one warmup (AITER's first call compiles), then time five steady requests. The mean is those five. Do not set `PYTORCH_ROCM_ARCH` on the `docker run` line.

## Run

One MI355X, Docker, `/dev/kfd`, and `/dev/dri`. Change into this directory first. The script mounts that directory at `/work`.

```bash
cd examples/cosmos3-mi355x-repro
export HF_TOKEN=...
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
./run_bench.sh
```

The script writes `obs.png`, `mstar-unprofiled.json`, and `omni-unprofiled.json` in the current directory.

`mstar-unprofiled.json` and `omni-unprofiled.json` report mean milliseconds and `action_rate_hz`. The server log should show `AiterFmhaAttentionManager` and `paged KV writes disabled`.

## What we measured

These are not a pass/fail gate. Absolute numbers can vary by node and software state.

One measured run on one AMD Instinct MI355X, unprofiled n=5 (warmup and the separate profiled request excluded):

- M*: mean 1107.73 ms, 14.444 actions/s
- vLLM-Omni: mean 816.10 ms, 19.605 actions/s

That run used this ROCm PyTorch stack with a request-profiler overlay present. The five steady requests were not profiled. This Dockerfile leaves that overlay out.

## Tracing

Chrome traces are not required for the timings above. To record M* GPU kernels, add both of these on top of this image: a request profiler that calls `torch.profiler` start and stop on the worker main thread, and a `site-packages` `.pth` that runs before torch import and executes `os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)`. A spawned worker inherits that variable from the parent after the parent imports torch, and Kineto then records no GPU events. vLLM-Omni traces are a profiler wrapped around `Omni.generate` in the client process.
