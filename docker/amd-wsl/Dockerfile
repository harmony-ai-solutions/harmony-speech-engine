# Dockerfile for Harmony-Speech-Engine AMD WSL2
# Uses PyTorch base image from Harmony AI quickstart repo
FROM harmonyai/base-pytorch-rocm-wsl2:2.7.1-latest

# Default ROCm ARCHes optimized for RDNA GPUs on WSL2
ARG PYTORCH_ROCM_ARCH="gfx1010;gfx1012;gfx1030;gfx1031;gfx1100;gfx1200;gfx1201"

# Set home dir
ENV HOME=/app/harmony-speech-engine
WORKDIR $HOME

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED=True

# Upgrade OS Packages + Prepare Python Environment
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y bzip2 g++ git make python3 python3-pip tzdata libeigen3-dev gcc curl libsndfile1 zlib1g-dev ffmpeg libmecab-dev ccache \
    && rm -fr /var/lib/apt/lists/*

# Alias python3 to python and upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python || true
RUN apt-get install --only-upgrade python3-pip

# Clone Repo and install requirements
RUN git clone https://github.com/harmony-ai-solutions/harmony-speech-engine.git $HOME
COPY entrypoint.sh docker/amd-wsl/entrypoint.sh
RUN chmod +x docker/amd-wsl/entrypoint.sh
# Install requirements but skip torch and torchaudio (already in base image)
RUN grep -v '^torch$' requirements-common.txt | grep -v '^torchaudio$' > requirements-common-notorch.txt && \
    echo "-r requirements-common-notorch.txt" > requirements-rocm-notorch.txt && \
    pip3 install -r requirements-rocm-notorch.txt --break-system-packages

# Download NLTK modules required by OpenVoice
RUN python -m nltk.downloader all

# Allow build servers to limit ninja build jobs. For reference
# see https://github.com/PygmalionAI/aphrodite-engine/wiki/1.-Installation#build-from-source
ARG MAX_JOBS
ENV MAX_JOBS=${MAX_JOBS}

# Export the ROCM variables correctly for WSL2
ENV LLVM_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer
ENV PATH=$PATH:/opt/rocm/bin:/libtorch/bin:
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib/:/libtorch/lib:
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/libtorch/include:/libtorch/include/torch/csrc/api/include/:/opt/rocm/include/:
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV CCACHE_DIR=/root/.cache/ccache

# Set target device for ROCm
#ENV HARMONYSPEECH_TARGET_DEVICE=rocm
# Use precompiled binaries to avoid build issues
#ENV HARMONYSPEECH_USE_PRECOMPILED=1

# Set Hugging Face cache directory
RUN mkdir -p /app/harmony-speech-engine/cache
ENV HF_HOME=/app/harmony-speech-engine/cache
ENV NUMBA_CACHE_DIR=${HF_HOME}/numba_cache
RUN python3 -m pip install --no-cache-dir --break-system-packages --no-build-isolation -e .

# Entrypoint exec form doesn't do variable substitution automatically ($HOME)
ENTRYPOINT ["/app/harmony-speech-engine/docker/amd-wsl/entrypoint.sh"]
EXPOSE 12080

# Service UID needs write access to $HOME to create temporary folders, see #458
RUN chown 1000:1000 ${HOME}
USER 1000:0
