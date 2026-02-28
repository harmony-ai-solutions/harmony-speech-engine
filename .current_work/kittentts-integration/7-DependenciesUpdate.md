# Phase 7: Dependencies Update

## Objective

Add KittenTTS dependencies to `requirements-common.txt` and the Dockerfiles. The new packages are `onnxruntime`, `misaki[en]`, `espeakng_loader`, `num2words`, and `spacy`.

## Analysis of Existing Dependencies

Checking against [`requirements-common.txt`](requirements-common.txt):
- `numpy` ✅ already present
- `soundfile` ✅ already present
- `huggingface_hub` ✅ already present

New packages needed (from KittenTTS `requirements.txt`):
- `onnxruntime` — ONNX Runtime CPU inference engine
- `misaki[en]` — phonemizer for English (`>=0.9.4`)
- `espeakng_loader` — loads espeak-ng for phonemization
- `num2words` — text normalization (numbers to words)
- `spacy` — NLP text preprocessing

## File to Modify

### `requirements-common.txt`

Add the following lines to `requirements-common.txt`:

```
# KittenTTS dependencies
onnxruntime
misaki[en]>=0.9.4
espeakng_loader
num2words
spacy
```

### Dockerfiles

The Dockerfiles install system packages. The `espeak-ng` system library needs to be installed for `espeakng_loader` and `misaki[en]` to work.

In **all three Dockerfiles** ([`docker/cpu/Dockerfile`](docker/cpu/Dockerfile), [`docker/nvidia/Dockerfile`](docker/nvidia/Dockerfile), [`docker/amd/Dockerfile`](docker/amd/Dockerfile)), add `espeak-ng` to the `apt-get install` line:

Current line (example from CPU Dockerfile):
```dockerfile
    && apt-get install -y bzip2 g++ git make python3 python3-pip tzdata libeigen3-dev gcc curl libsndfile1 zlib1g-dev ffmpeg libmecab-dev ccache \
```

Updated line:
```dockerfile
    && apt-get install -y bzip2 g++ git make python3 python3-pip tzdata libeigen3-dev gcc curl libsndfile1 zlib1g-dev ffmpeg libmecab-dev ccache espeak-ng \
```

The `docker/amd-wsl/Dockerfile` also needs this change if it exists.

## Notes

- `onnxruntime` (CPU version) is appropriate here. The GPU variant would be `onnxruntime-gpu`, but KittenTTS is designed for CPU inference, so `onnxruntime` is correct for all Dockerfiles including NVIDIA/AMD ones.
- `spacy` requires a language model (`en_core_web_sm`) to be downloaded at runtime. However, looking at [`kittentts/preprocess.py`](`.current_work/KittenTTS/kittentts/preprocess.py`), spacy is used for text preprocessing. Check whether a spacy model download step is needed in the Dockerfile (similar to the `python -m nltk.downloader all` step). This should be verified during implementation.
- `misaki[en]` requires `espeak-ng` system library — hence the Dockerfile change.

## Progress Checklist

- [ ] Add `onnxruntime`, `misaki[en]>=0.9.4`, `espeakng_loader`, `num2words`, `spacy` to [`requirements-common.txt`](requirements-common.txt)
- [ ] Add `espeak-ng` to apt-get install in [`docker/cpu/Dockerfile`](docker/cpu/Dockerfile)
- [ ] Add `espeak-ng` to apt-get install in [`docker/nvidia/Dockerfile`](docker/nvidia/Dockerfile)
- [ ] Add `espeak-ng` to apt-get install in [`docker/amd/Dockerfile`](docker/amd/Dockerfile)
- [ ] Check if spacy model download step is needed in Dockerfiles
