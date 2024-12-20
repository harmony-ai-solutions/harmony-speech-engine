# Dockerfile for Harmony-Speech-Engine
# mostly copied over from from PygmalionAI / Aphrodite Engine
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set home dir
ENV HOME /app/harmony-speech-engine
WORKDIR $HOME

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Upgrade OS Packages + Prepare Python Environment
RUN set -eux; \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y bzip2 g++ git make python3-pip tzdata libeigen3-dev gcc curl libsndfile1 zlib1g-dev ffmpeg \
    && rm -fr /var/lib/apt/lists/*

# Alias python3 to python and upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Clone Repo and install requirements
RUN git clone https://github.com/harmony-ai-solutions/harmony-speech-engine.git $HOME \
    && chmod +x docker/entrypoint.sh
RUN pip3 install -r requirements.txt

# Download NLTK modules required by OpenVoice
RUN python -m nltk.downloader all

# Allow build servers to limit ninja build jobs. For reference
# see https://github.com/PygmalionAI/aphrodite-engine/wiki/1.-Installation#build-from-source
ARG MAX_JOBS
ENV MAX_JOBS=${MAX_JOBS}

# Export the CUDA_HOME variable correctly
ENV CUDA_HOME=/usr/local/cuda

# Set Hugging Face cache directory
RUN mkdir -p /app/harmony-speech-engine/cache
ENV HF_HOME=/app/harmony-speech-engine/cache

ENV NUMBA_CACHE_DIR=$HF_HOME/numba_cache
ENV TORCH_CUDA_ARCH_LIST="6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX"
RUN python3 -m pip install --no-cache-dir -e .

# Workaround to properly install flash-attn. For reference => Not needed for Speech Engine currently
# see: https://github.com/Dao-AILab/flash-attention/issues/453
# RUN python3 -m pip install 'flash-attn>=2.5.8' --no-build-isolation

# Entrypoint exec form doesn't do variable substitution automatically ($HOME)
ENTRYPOINT ["/app/harmony-speech-engine/docker/entrypoint.sh"]
EXPOSE 12080

# Service UID needs write access to $HOME to create temporary folders, see #458
RUN chown 1000:1000 ${HOME}
USER 1000:0