# RunPod pod image for SignRot/QuaRot experiments.
#
# Always targets linux/amd64 + CUDA 12.8 regardless of the build machine
# (both FROM lines are platform-pinned; no GPU is needed at build time).
#
# Build (from repo root, with submodules checked out):
#   git submodule update --init --recursive
#   docker build -t <dockerhub-user>/signrot:cu128 .
#   docker push <dockerhub-user>/signrot:cu128
#
# On Apple Silicon the amd64 build runs under Rosetta emulation: enable
# "Use Rosetta for x86_64/amd64 emulation" in Docker Desktop, give it enough
# RAM, and cap nvcc parallelism with --build-arg MAX_JOBS=4 if it OOMs.
#
# Supported GPUs are fixed by the gencode flags in setup.py: sm_75/80/86/90
# (T4, A100, A40/A6000, L40S/4090, H100/H200). Blackwell (5090/B200) needs an
# extra gencode entry in setup.py and fast-hadamard-transform.

FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:0.11.2 AS uv

FROM --platform=linux/amd64 nvidia/cuda:12.8.1-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git cmake ninja-build openssh-server ca-certificates curl vim tmux htop && \
    rm -rf /var/lib/apt/lists/*

# Pin uv to the version used locally so lock resolution/build behavior matches
COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /app

# Dependencies + fast-hadamard-transform kernel build. This is the slow layer
# (torch cu128, flash-attn wheel, fht nvcc build) and only rebuilds when the
# lock or the submodule changes — not on every code edit.
# Cap parallel nvcc jobs (defaults to all cores); lower this on emulated builds
ARG MAX_JOBS

COPY pyproject.toml uv.lock .python-version ./
COPY third-party/fast-hadamard-transform third-party/fast-hadamard-transform
RUN --mount=type=cache,target=/root/.cache/uv \
    MAX_JOBS=${MAX_JOBS:-$(nproc)} uv sync --frozen --no-install-project

# Full source; builds the signrot CUDA kernels
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    MAX_JOBS=${MAX_JOBS:-$(nproc)} uv sync --frozen

ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH

RUN chmod +x /app/docker/start.sh
EXPOSE 22
CMD ["/app/docker/start.sh"]
