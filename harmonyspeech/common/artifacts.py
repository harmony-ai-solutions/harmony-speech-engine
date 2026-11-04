"""Typed in-process pipeline artifacts (Phase 2: artifact bus).

Pipeline steps exchange these objects instead of API-shaped strings. Nothing in
here serializes anything: wire text/JSON/WAV encoding stays at the HTTP boundary
(``harmonyspeech/endpoints/openai/wire.py``).

**Waveform representation decision (2-1).** Audio is standardized on
``torch.Tensor``, shape ``[1, samples]``, ``float32``. The rationale:

* torch-native steps (Chatterbox, OpenVoice, VoiceFixer, the vocoders) avoid a
  numpy<->tensor round trip on every hop;
* CPU-only steps (faster-whisper, silero, librosa preprocessing) can get a
  zero-copy numpy view with :attr:`AudioArtifact.numpy` —
  ``torch.from_numpy`` shares memory with the numpy buffer, so
  :meth:`AudioArtifact.from_numpy` is cheap;
* a single representation means the device-move rule below has exactly one
  implementation.

**Device discipline.** Every artifact carries ``device`` and every cross-device
move is explicit (:meth:`AudioArtifact.to`). ``to()`` is a no-op when the target
matches and logs at debug level when it actually copies, which makes accidental
per-hop transfers visible in Phase 2-3 reviews. Artifacts never move implicitly
as a side effect of being read.

**Ownership.** Artifacts are owned by
:class:`harmonyspeech.common.request.EngineRequest` (``artifacts`` dict, keyed by
request-field name). When a request finishes, the scheduler calls
``EngineRequest.clear_artifacts()`` so tensor references (and their GPU memory)
are released deterministically rather than "whenever the request object happens
to be collected".
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import torch
from loguru import logger

from harmonyspeech.common import audio_codec

__all__ = [
    "AudioArtifact",
    "MelArtifact",
    "SpeakerArtifact",
    "VadArtifact",
    "SpeakerBackend",
]


SpeakerBackend = Literal["openvoice", "chatterbox", "harmonyspeech"]


def _same_device(current: torch.device | None, target: torch.device) -> bool:
    """Compare devices, treating ``cuda`` as ``cuda:0``-equivalent."""
    if current is None:
        return False
    current = torch.device(current)
    if current.type != target.type:
        return False
    return target.index is None or current.index == target.index


def _as_float_tensor(value: Any) -> torch.Tensor:
    """Return ``value`` as a float32 tensor (copies only when needed)."""
    if torch.is_tensor(value):
        return value.detach().to(dtype=torch.float32)
    return torch.as_tensor(np.asarray(value, dtype=np.float32))


def _as_batched_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Normalize a waveform to ``[1, samples]`` float32."""
    tensor = _as_float_tensor(waveform)
    if tensor.ndim == 0:
        raise ValueError("AudioArtifact waveform must have at least one dimension")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 2:
        # Collapse extra leading axes (e.g. [1, 1, samples] from a vocoder).
        tensor = tensor.reshape(-1, tensor.shape[-1])
        if tensor.shape[0] != 1:
            tensor = tensor.mean(dim=0, keepdim=True)
    return tensor.contiguous()


@dataclass
class AudioArtifact:
    """A mono waveform.

    Attributes:
        waveform: ``float32`` tensor shaped ``[1, samples]``.
        sample_rate: samples per second of ``waveform``.
        device: device the waveform currently lives on. Always populated (it is
            inferred from ``waveform`` when not given).
    """

    waveform: torch.Tensor
    sample_rate: int
    device: torch.device | None = None

    def __post_init__(self) -> None:
        self.sample_rate = int(self.sample_rate)
        if self.sample_rate <= 0:
            raise ValueError(f"AudioArtifact sample_rate must be positive, got {self.sample_rate!r}")
        self.waveform = _as_batched_waveform(self.waveform)
        if self.device is None:
            self.device = self.waveform.device
        else:
            self.device = torch.device(self.device)
            if not _same_device(self.device, self.waveform.device):
                raise ValueError(
                    f"AudioArtifact device mismatch: device={self.device} but waveform is on "
                    f"{self.waveform.device}. Move the waveform explicitly instead."
                )
            self.device = self.waveform.device

    # -- constructors ---------------------------------------------------
    @classmethod
    def from_numpy(
        cls, waveform: np.ndarray, sample_rate: int, device: str | torch.device | None = None
    ) -> AudioArtifact:
        """Build an artifact from a numpy waveform (shares memory when contiguous)."""
        array = np.ascontiguousarray(np.asarray(waveform, dtype=np.float32))
        tensor = torch.from_numpy(array)
        return cls(waveform=tensor, sample_rate=sample_rate, device=device)

    # -- accessors ------------------------------------------------------
    @property
    def numpy(self) -> np.ndarray:
        """Mono ``float32`` numpy view (``[samples]``), copied to CPU if needed."""
        return self.waveform.detach().cpu().numpy().reshape(-1)

    @property
    def num_samples(self) -> int:
        return int(self.waveform.shape[-1])

    @property
    def duration_s(self) -> float:
        return self.num_samples / float(self.sample_rate)

    def resampled(self, sample_rate: int | None) -> AudioArtifact:
        """Return a transient, resampled view without caching it on the request.

        Caching resampled copies per (field, rate) would multiply memory; steps
        ask for the rate they need and the result is short lived.
        """
        if sample_rate is None or int(sample_rate) == self.sample_rate:
            return self
        resampled = audio_codec.resample(self.numpy, self.sample_rate, int(sample_rate))
        return AudioArtifact.from_numpy(resampled, int(sample_rate))

    def to_wav_bytes(self) -> bytes:
        """Encode the artifact as a PCM-16 WAV blob (boundary/3rd-party adapters)."""
        return audio_codec.encode_audio(self.numpy, self.sample_rate, "wav")

    # -- device discipline ---------------------------------------------
    def to(self, device: str | torch.device) -> AudioArtifact:
        """Explicitly move the waveform. No-op when already on ``device``."""
        target = torch.device(device)
        if _same_device(self.device, target):
            return self
        logger.debug(
            f"AudioArtifact: moving waveform {tuple(self.waveform.shape)} {self.device} -> {target}"
        )
        waveform = self.waveform.to(target)
        return replace(self, waveform=waveform, device=waveform.device)

    def __len__(self) -> int:
        return self.num_samples


@dataclass
class MelArtifact:
    """A mel/mel-like spectrogram tensor.

    ``mel`` is ``float32`` and shaped ``[1, mel_bins, frames]`` for the canonical
    backends (HarmonySpeech synth->vocoder). VoiceFixer's restorer emits a
    channel/batch/frame/bin tensor; the handler stores ``[:, 0]`` transposed so
    the canonical ``[1, bins, frames]`` contract holds (the vocoder only ever
    consumes channel 0, so this is numerically identical).

    ``breaks`` carries the per-chunk frame counts the HarmonySpeech vocoder uses
    to re-insert inter-chunk silence; it travels alongside the mel so the pairing
    can never be lost or reordered.
    """

    mel: torch.Tensor
    sample_rate: int
    hop_length: int
    device: torch.device | None = None
    breaks: list[int] | None = None

    def __post_init__(self) -> None:
        self.sample_rate = int(self.sample_rate)
        self.hop_length = int(self.hop_length)
        tensor = _as_float_tensor(self.mel)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"MelArtifact.mel must be [1, mel_bins, frames], got shape {tuple(tensor.shape)}")
        self.mel = tensor.contiguous()
        if self.device is None:
            self.device = self.mel.device
        else:
            self.device = torch.device(self.device)
            if not _same_device(self.device, self.mel.device):
                raise ValueError(
                    f"MelArtifact device mismatch: device={self.device} but mel is on {self.mel.device}"
                )
            self.device = self.mel.device
        if self.breaks is not None:
            self.breaks = [int(value) for value in self.breaks]

    @classmethod
    def from_numpy(
        cls,
        mel: np.ndarray,
        sample_rate: int,
        hop_length: int,
        breaks: list[int] | None = None,
        device: str | torch.device | None = None,
    ) -> MelArtifact:
        array = np.ascontiguousarray(np.asarray(mel, dtype=np.float32))
        return cls(
            mel=torch.from_numpy(array),
            sample_rate=sample_rate,
            hop_length=hop_length,
            device=device,
            breaks=breaks,
        )

    @property
    def numpy(self) -> np.ndarray:
        return self.mel.detach().cpu().numpy()

    @property
    def mel_bins(self) -> int:
        return int(self.mel.shape[1])

    @property
    def frames(self) -> int:
        return int(self.mel.shape[-1])

    def to(self, device: str | torch.device) -> MelArtifact:
        target = torch.device(device)
        if _same_device(self.device, target):
            return self
        logger.debug(f"MelArtifact: moving mel {tuple(self.mel.shape)} {self.device} -> {target}")
        mel = self.mel.to(target)
        return replace(self, mel=mel, device=mel.device)


@dataclass
class SpeakerArtifact:
    """Speaker conditioning, backend-polymorphic.

    * ``embedding`` — OpenVoice / HarmonySpeech speaker embedding tensor.
    * ``conditionals`` — Chatterbox ``Conditionals`` object, opaque to the bus.
      It is passed **by reference**: the consuming handler moves it to its own
      device (``conditionals.to(device)``). ``SpeakerArtifact.to()`` deliberately
      leaves it untouched so the bus never hides a cross-device copy.
    * ``reference_audio`` — the raw reference clip a downstream step must turn
      into conditionals itself (Chatterbox ``prepare_conditionals``).
    """

    embedding: torch.Tensor | None = None
    conditionals: Any | None = None
    backend: SpeakerBackend | None = None
    device: torch.device | None = None
    reference_audio: AudioArtifact | None = None

    def __post_init__(self) -> None:
        if self.embedding is not None:
            self.embedding = _as_float_tensor(self.embedding)
        if self.device is None and self.embedding is not None:
            self.device = self.embedding.device
        elif self.device is not None:
            self.device = torch.device(self.device)

    def to(self, device: str | torch.device) -> SpeakerArtifact:
        """Move the embedding (and reference audio). Conditionals stay by reference."""
        target = torch.device(device)
        embedding = self.embedding
        if embedding is not None and not _same_device(embedding.device, target):
            logger.debug(f"SpeakerArtifact: moving embedding {tuple(embedding.shape)} -> {target}")
            embedding = embedding.to(target)
        reference_audio = self.reference_audio
        if reference_audio is not None:
            reference_audio = reference_audio.to(target)
        device_value = self.device
        if embedding is not None:
            device_value = embedding.device
        elif reference_audio is not None:
            device_value = reference_audio.device
        return replace(
            self,
            embedding=embedding,
            reference_audio=reference_audio,
            device=device_value,
        )


@dataclass
class VadArtifact:
    """VAD/transcription segments (whisper segment dicts or silero timestamps).

    ``info`` and ``text`` are the whisper extras the tone-converter encoder does
    not need but transcription consumers do; Phase 3-4 normalizes silero output
    to the same schema.
    """

    segments: list[dict] = field(default_factory=list)
    info: dict | None = None
    text: str | None = None
    device: torch.device | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> VadArtifact:
        """Build from the whisper JSON payload (``{"text", "segments", "info"}``)."""
        segments = payload.get("segments") or []
        if not isinstance(segments, list):
            raise ValueError("VAD payload 'segments' must be a list")
        info = payload.get("info")
        if info is not None and not isinstance(info, dict):
            # whisper's ``info`` is serialized with dataclasses.asdict, but accept objects too.
            info = getattr(info, "__dict__", None)
        return cls(segments=list(segments), info=info, text=payload.get("text"))

    @property
    def has_speech(self) -> bool:
        return len(self.segments) > 0

    def to(self, device: str | torch.device) -> VadArtifact:
        """No tensors to move; kept for a uniform artifact interface."""
        return self
