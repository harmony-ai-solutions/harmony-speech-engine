"""Phase 14-2: byte-level streaming primitives.

Pins the pieces both streaming envelopes share, without any FastAPI or engine
involvement:

* :func:`wav_stream_header` — golden hex fixture plus a field-by-field RIFF
  decode, and the documented stdlib-``wave`` duration wart;
* :func:`pcm16_frame` — the one canonical s16le conversion (artifact / numpy /
  container bytes / base64), including the codec-layer round trip;
* :class:`StreamResampler` — the continuity gate: a sine sweep split at
  arbitrary (odd, non-block-aligned, 0/1-sample) boundaries must match one-shot
  ``soxr.resample`` of the concatenation within ``rtol=1e-6``, with no gaps or
  duplicates, plus emitted-sample accounting;
* the SSE primitives — frame shape, strict-parser escaping, the ``audio`` (not
  ``delta``) field, usage extension, error shape and b64 round-trip.
"""

import base64
import inspect
import io
import json
import struct
import wave

import numpy as np
import pytest
import soundfile as sf

from harmonyspeech.common import audio_codec
from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.endpoints.openai.streaming import (
    SSE_DONE_FRAME,
    SSE_EVENT_DELTA,
    SSE_EVENT_DONE,
    SSE_EVENT_ERROR,
    SSE_KEEPALIVE,
    WAV_STREAM_HEADER_BYTES,
    ResampleStream,
    StreamResampler,
    delta_payload,
    done_payload,
    error_payload,
    keepalive,
    pcm16_frame,
    sse_event,
    wav_stream_header,
)

pytestmark = pytest.mark.unit

#: Golden fixture: 24 kHz mono s16le streaming header, byte for byte.
GOLDEN_HEADER_24K_MONO_S16 = (
    "52494646ffffffff57415645666d74201000000001000100c05d000080bb00000200100064617461ffffffff"
)

#: A small waveform whose float32 -> s16le -> WAV -> s16le round trip is exact
#: (0.5 / -0.5 land on 16384 / -16384, which re-encode unchanged).
ROUND_TRIP_WAVE = np.array([0.0, 0.25, -0.25, 0.5, -0.5], dtype=np.float32)


def _parse_frame(frame: str) -> tuple[str, dict]:
    """Strict SSE line parser: one ``event:`` line, one ``data:`` line, blank."""
    assert frame.endswith("\n\n")
    lines = frame[:-1].split("\n")
    assert len(lines) == 3, f"frame is not exactly event + data + blank: {frame!r}"
    event_line, data_line, blank = lines
    assert blank == ""
    assert event_line.startswith("event: ")
    assert data_line.startswith("data: ")
    return event_line[len("event: ") :], json.loads(data_line[len("data: ") :])


def _sine_sweep(rate: int, seconds: float = 0.5, f0: float = 100.0, f1: float = 6000.0) -> np.ndarray:
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    phase = 2.0 * np.pi * (f0 * t + (f1 - f0) / (2.0 * seconds) * t * t)
    return (0.5 * np.sin(phase)).astype(np.float32)


def _stream_in_splits(resampler: StreamResampler, samples: np.ndarray, splits) -> np.ndarray:
    pieces = []
    previous = 0
    for split in list(splits) + [len(samples)]:
        pieces.append(resampler.feed(samples[previous:split]))
        previous = split
    pieces.append(resampler.flush())
    return np.concatenate([piece for piece in pieces if piece.size])


# ---------------------------------------------------------------------------
# WAV streaming header
# ---------------------------------------------------------------------------


class TestWavStreamHeader:
    def test_golden_hex_fixture(self):
        assert wav_stream_header(24000).hex() == GOLDEN_HEADER_24K_MONO_S16

    def test_length_is_the_riff_pcm_header(self):
        assert WAV_STREAM_HEADER_BYTES == 44
        assert len(wav_stream_header(24000)) == WAV_STREAM_HEADER_BYTES

    def test_fields_field_by_field(self):
        data = wav_stream_header(24000)

        assert data[0:4] == b"RIFF"
        assert struct.unpack_from("<I", data, 4)[0] == 0xFFFFFFFF
        assert data[8:12] == b"WAVE"

        assert data[12:16] == b"fmt "
        fmt_size, audio_format, channels, rate, byte_rate, block_align, bits = struct.unpack_from(
            "<IHHIIHH", data, 16
        )
        assert fmt_size == 16
        assert audio_format == 1  # PCM
        assert channels == 1
        assert rate == 24000
        assert byte_rate == 24000 * 1 * 16 // 8
        assert block_align == 1 * 16 // 8
        assert bits == 16

        assert data[36:40] == b"data"
        assert struct.unpack_from("<I", data, 40)[0] == 0xFFFFFFFF

    @pytest.mark.parametrize(
        ("rate", "channels", "bits"),
        [(8000, 1, 16), (16000, 1, 16), (24000, 1, 16), (44100, 2, 16), (48000, 2, 24)],
    )
    def test_byte_rate_and_block_align_arithmetic(self, rate, channels, bits):
        data = wav_stream_header(rate, channels, bits)
        _fmt_size, _audio_format, got_channels, got_rate, byte_rate, block_align, got_bits = (
            struct.unpack_from("<IHHIIHH", data, 16)
        )
        assert got_channels == channels
        assert got_rate == rate
        assert got_bits == bits
        assert block_align == channels * bits // 8
        assert byte_rate == rate * channels * bits // 8

    def test_both_size_fields_are_the_streaming_sentinel(self):
        data = wav_stream_header(16000)
        assert struct.unpack_from("<I", data, 4)[0] == 0xFFFFFFFF
        assert struct.unpack_from("<I", data, 40)[0] == 0xFFFFFFFF

    @pytest.mark.parametrize("rate", [0, -1, 1.5, "24000", True, None])
    def test_rejects_invalid_sample_rate(self, rate):
        with pytest.raises(ValueError, match="sample_rate"):
            wav_stream_header(rate)

    @pytest.mark.parametrize("channels", [0, -1, 70000, 1.5, "1"])
    def test_rejects_invalid_channels(self, channels):
        with pytest.raises(ValueError, match="channels"):
            wav_stream_header(24000, channels)

    @pytest.mark.parametrize("bits", [0, 12, 64, "16"])
    def test_rejects_invalid_bits(self, bits):
        with pytest.raises(ValueError, match="bits"):
            wav_stream_header(24000, 1, bits)

    def test_soundfile_reads_the_stream_to_eof_but_stdlib_wave_misreports(self):
        """The documented wart: ``wave`` trusts the 0xFFFFFFFF data size."""
        samples = _sine_sweep(24000, seconds=0.1, f1=2000.0)
        blob = wav_stream_header(24000) + pcm16_frame(samples)

        with sf.SoundFile(io.BytesIO(blob)) as handle:
            assert handle.samplerate == 24000
            assert handle.channels == 1
            assert handle.frames == len(samples)
            decoded = handle.read(dtype="float32")
        np.testing.assert_allclose(decoded, samples, atol=1e-4)

        with wave.open(io.BytesIO(blob)) as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 24000
            # 0xFFFFFFFF bytes / 2 bytes per frame — the wart, pinned on purpose.
            assert handle.getnframes() == 0xFFFFFFFF // 2


# ---------------------------------------------------------------------------
# PCM frames
# ---------------------------------------------------------------------------


class TestPcm16Frame:
    def test_float32_to_s16le_golden(self):
        samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5], dtype=np.float32)
        pcm = pcm16_frame(samples)
        assert pcm == bytes.fromhex("0000004000c0ff7f0180ff7f0180")
        assert np.frombuffer(pcm, dtype="<i2").tolist() == [
            0,
            16384,
            -16384,
            32767,
            -32767,
            32767,  # +1.5 clipped
            -32767,  # -1.5 clipped
        ]

    def test_accepts_the_typed_audio_artifact(self):
        artifact = AudioArtifact.from_numpy(ROUND_TRIP_WAVE, 24000)
        assert pcm16_frame(artifact) == pcm16_frame(ROUND_TRIP_WAVE)
        # The artifact's rate is only used for a resample, which "pcm" never does.
        assert pcm16_frame(artifact, 8000) == pcm16_frame(ROUND_TRIP_WAVE)

    @pytest.mark.parametrize("shape", [(1, -1), (-1, 1)])
    def test_accepts_two_dimensional_mono(self, shape):
        reshaped = ROUND_TRIP_WAVE.reshape(shape)
        assert pcm16_frame(reshaped) == pcm16_frame(ROUND_TRIP_WAVE)

    def test_downmixes_stereo_like_encode_audio(self):
        stereo = np.stack([ROUND_TRIP_WAVE, ROUND_TRIP_WAVE], axis=1)
        assert pcm16_frame(stereo) == pcm16_frame(ROUND_TRIP_WAVE)

    def test_container_bytes_go_through_the_codec_decoder(self):
        wav = audio_codec.encode_audio(ROUND_TRIP_WAVE, 24000, "wav")
        assert pcm16_frame(wav) == pcm16_frame(ROUND_TRIP_WAVE)

    def test_legacy_base64_payload_is_decoded_once(self):
        wav = audio_codec.encode_audio(ROUND_TRIP_WAVE, 24000, "wav")
        legacy = base64.b64encode(wav).decode("ascii")
        assert pcm16_frame(legacy) == pcm16_frame(wav) == pcm16_frame(ROUND_TRIP_WAVE)

    def test_empty_waveform_is_an_empty_frame(self):
        assert pcm16_frame(np.empty(0, dtype=np.float32)) == b""

    def test_rejects_non_audio_bytes(self):
        with pytest.raises(audio_codec.AudioDecodeError):
            pcm16_frame(b"definitely not audio")

    def test_rejects_three_dimensional_input(self):
        with pytest.raises(ValueError, match="mono waveform"):
            pcm16_frame(np.zeros((2, 2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# Stateful resampler
# ---------------------------------------------------------------------------

#: Named split strategies, relative to the sweep length. Includes a split at
#: sample 0 and 1, odd sizes that are not block-aligned, and an empty tail.
SPLIT_SPECS = {
    "no_split": lambda n: [],
    "split_at_zero": lambda n: [0],
    "split_at_one": lambda n: [1],
    "split_at_three": lambda n: [3],
    "odd_midpoint": lambda n: [n // 2 + 1],
    "many_odd_sizes": lambda n: [0, 1, 2, 3, 7, 17, 100, 1000, 4097],
    "last_sample": lambda n: [n - 1],
    "empty_tail_chunk": lambda n: [n],
}


class TestStreamResampler:
    def test_installed_soxr_exposes_the_pinned_resample_stream_api(self):
        """Pins the environment assumption recorded in 14-2's verification log."""
        import soxr

        assert isinstance(soxr.__version__, str) and soxr.__version__
        assert hasattr(soxr, "ResampleStream")

        init_params = inspect.signature(soxr.ResampleStream.__init__).parameters
        assert list(init_params) == [
            "self",
            "in_rate",
            "out_rate",
            "num_channels",
            "dtype",
            "quality",
        ]
        assert init_params["dtype"].default == "float32"
        assert init_params["quality"].default == "HQ"

        chunk_params = inspect.signature(soxr.ResampleStream.resample_chunk).parameters
        assert list(chunk_params) == ["self", "x", "last"]
        assert chunk_params["last"].default is False

    @pytest.mark.parametrize(
        ("in_rate", "out_rate"),
        [(48000, 16000), (44100, 8000), (24000, 22050), (16000, 48000)],
    )
    @pytest.mark.parametrize("spec", list(SPLIT_SPECS), ids=list(SPLIT_SPECS))
    def test_streamed_output_matches_one_shot_resample(self, in_rate, out_rate, spec):
        import soxr

        sweep = _sine_sweep(in_rate)
        one_shot = soxr.resample(sweep, in_rate, out_rate, quality="HQ")

        resampler = StreamResampler(in_rate, out_rate)
        streamed = _stream_in_splits(resampler, sweep, SPLIT_SPECS[spec](len(sweep)))

        # No gaps and no duplicates: identical length and sample values.
        assert len(streamed) == len(one_shot)
        np.testing.assert_allclose(streamed, one_shot, rtol=1e-6, atol=0.0)
        assert resampler.emitted_samples == len(streamed)
        assert resampler.input_samples == len(sweep)
        assert resampler.closed

    def test_emitted_sample_accounting_counts_feed_and_flush(self):
        sweep = _sine_sweep(48000)
        resampler = StreamResampler(48000, 16000)
        first = resampler.feed(sweep[:1000])
        second = resampler.feed(sweep[1000:])
        tail = resampler.flush()

        assert resampler.emitted_samples == len(first) + len(second) + len(tail)
        assert resampler.input_samples == len(sweep)
        assert resampler.flush().size == 0
        assert resampler.emitted_samples == len(first) + len(second) + len(tail)

    def test_equal_rates_are_a_pass_through(self):
        sweep = _sine_sweep(24000, seconds=0.1)
        resampler = StreamResampler(24000, 24000)
        output = resampler.feed(sweep)
        assert np.array_equal(output, sweep)
        assert resampler.flush().size == 0
        assert resampler.emitted_samples == len(sweep)
        assert resampler.closed

    def test_empty_chunks_are_ignored(self):
        sweep = _sine_sweep(48000, seconds=0.1)
        resampler = StreamResampler(48000, 16000)
        assert resampler.feed(np.empty(0, dtype=np.float32)).size == 0
        assert resampler.input_samples == 0
        assert resampler.emitted_samples == 0
        output = resampler.feed(sweep)
        tail = resampler.flush()
        assert resampler.emitted_samples == len(output) + len(tail)

    def test_mono_two_dimensional_chunks_are_accepted(self):
        sweep = _sine_sweep(48000, seconds=0.1)
        flat = StreamResampler(48000, 16000)
        flat_output = _stream_in_splits(flat, sweep, [100, 2000])

        matrix = StreamResampler(48000, 16000)
        matrix_output = _stream_in_splits(matrix, sweep.reshape(-1, 1), [100, 2000])

        assert np.array_equal(flat_output, matrix_output)

    def test_feed_after_flush_raises(self):
        resampler = StreamResampler(48000, 16000)
        resampler.feed(_sine_sweep(48000, seconds=0.05))
        resampler.flush()
        with pytest.raises(RuntimeError, match="after flush"):
            resampler.feed(np.zeros(10, dtype=np.float32))

    def test_flush_before_any_feed_returns_empty(self):
        resampler = StreamResampler(48000, 16000)
        assert resampler.flush().size == 0
        assert resampler.emitted_samples == 0
        assert resampler.closed

    def test_wrong_channel_count_raises(self):
        mono = StreamResampler(48000, 16000)
        with pytest.raises(ValueError, match="mono"):
            mono.feed(np.zeros((10, 2), dtype=np.float32))

        stereo = StreamResampler(48000, 16000, channels=2)
        with pytest.raises(ValueError, match="frames, 2"):
            stereo.feed(np.zeros(10, dtype=np.float32))

    @pytest.mark.parametrize(
        ("in_rate", "out_rate"),
        [(0, 16000), (48000, 0), (-1, 16000), (48000, -1), ("48000", 16000), (True, 16000)],
    )
    def test_rejects_invalid_rates(self, in_rate, out_rate):
        with pytest.raises(ValueError, match="rate"):
            StreamResampler(in_rate, out_rate)

    def test_rejects_invalid_channels(self):
        with pytest.raises(ValueError, match="channels"):
            StreamResampler(48000, 16000, channels=0)

    def test_resample_stream_alias_is_the_wrapper(self):
        assert ResampleStream is StreamResampler


# ---------------------------------------------------------------------------
# SSE primitives
# ---------------------------------------------------------------------------


class TestSseEvent:
    def test_frame_is_event_line_plus_one_data_line_plus_blank(self):
        frame = sse_event(SSE_EVENT_DELTA, delta_payload("AAAA", 0))
        assert frame == (
            'event: speech.audio.delta\ndata: {"type":"speech.audio.delta","audio":"AAAA","chunk_index":0}\n\n'
        )
        event, payload = _parse_frame(frame)
        assert event == SSE_EVENT_DELTA
        assert payload["audio"] == "AAAA"

    def test_newlines_and_quotes_are_escaped_and_stay_parseable(self):
        payload = {
            "type": SSE_EVENT_DELTA,
            "text": 'line1\nline2\r\n"quoted" \\ backslash',
            "unicode": "héllo 😀",
        }
        frame = sse_event(SSE_EVENT_DELTA, payload)

        # The JSON escapes the line breaks, so the frame never grows a second
        # data line and a strict line parser still recovers the payload.
        event, parsed = _parse_frame(frame)
        assert event == SSE_EVENT_DELTA
        assert parsed == payload
        assert "\\n" in frame and "\\r" in frame
        assert '"quoted"' not in frame  # quotes are escaped inside the JSON string
        assert "héllo 😀" in frame  # ensure_ascii=False keeps UTF-8 readable

    @pytest.mark.parametrize("event_type", ["bad\nevent", "bad\revent"])
    def test_rejects_multiline_event_type(self, event_type):
        with pytest.raises(ValueError, match="single line"):
            sse_event(event_type, {})

    def test_rejects_empty_event_type(self):
        with pytest.raises(ValueError, match="empty"):
            sse_event("", {})

    def test_rejects_non_json_floats(self):
        with pytest.raises(ValueError):
            sse_event(SSE_EVENT_DELTA, {"duration_s": float("nan")})

    def test_keepalive_is_a_comment_frame(self):
        assert keepalive() == ": keepalive\n\n" == SSE_KEEPALIVE
        assert keepalive().startswith(":")

    def test_done_frame_is_a_separate_data_only_emission(self):
        assert SSE_DONE_FRAME == "data: [DONE]\n\n"
        lines = SSE_DONE_FRAME[:-1].split("\n")
        assert lines == ["data: [DONE]", ""]


class TestSsePayloads:
    def test_delta_payload_uses_audio_not_delta(self):
        payload = delta_payload("QUJD", 7)
        assert payload == {
            "type": SSE_EVENT_DELTA,
            "audio": "QUJD",
            "chunk_index": 7,
        }
        assert "delta" not in payload

    def test_delta_payload_rejects_negative_chunk_index(self):
        with pytest.raises(ValueError, match="chunk_index"):
            delta_payload("QUJD", -1)

    def test_delta_audio_b64_round_trip(self):
        samples = ROUND_TRIP_WAVE
        pcm = pcm16_frame(samples)
        payload = delta_payload(base64.b64encode(pcm).decode("ascii"), 0)
        assert base64.b64decode(payload["audio"]) == pcm
        assert np.frombuffer(base64.b64decode(payload["audio"]), dtype="<i2").tolist() == [
            0,
            8192,
            -8192,
            16384,
            -16384,
        ]

    def test_done_payload_fields_and_additive_usage(self):
        payload = done_payload(
            id="tts-abc",
            model="chatterbox",
            created=1700000000,
            sample_rate=24000,
            format="pcm",
            chunks=3,
            duration_s=1.5,
            usage={"input_characters": 42},
        )
        assert payload == {
            "type": SSE_EVENT_DONE,
            "id": "tts-abc",
            "model": "chatterbox",
            "created": 1700000000,
            "sample_rate": 24000,
            "format": "pcm",
            "chunks": 3,
            "duration_s": 1.5,
            "usage": {"input_characters": 42, "audio_seconds": 1.5},
        }

    def test_done_payload_does_not_mutate_the_usage_argument(self):
        usage = {"input_characters": 5}
        done_payload("id", "model", 0, 24000, "wav", 1, 0.5, usage)
        assert usage == {"input_characters": 5}

    def test_done_payload_without_usage_defaults_input_characters(self):
        payload = done_payload("id", "model", 0, 16000, "pcm", 1, 2 / 3)
        assert payload["usage"] == {"audio_seconds": 0.666667, "input_characters": 0}
        assert payload["duration_s"] == 0.666667

    def test_done_payload_keeps_foreign_usage_keys(self):
        payload = done_payload(
            "id", "model", 0, 24000, "pcm", 2, 1.0, {"type": "duration", "seconds": 1}
        )
        assert payload["usage"] == {
            "type": "duration",
            "seconds": 1,
            "audio_seconds": 1.0,
            "input_characters": 0,
        }

    def test_error_payload_from_exception(self):
        assert error_payload(ValueError("bad input")) == {
            "error": {"message": "bad input", "type": "ValueError", "code": None}
        }

    def test_error_payload_uses_a_code_attribute_when_present(self):
        class RateLimitedError(Exception):
            code = "rate_limited"

        assert error_payload(RateLimitedError("slow down"))["error"]["code"] == "rate_limited"

    def test_error_payload_falls_back_to_the_class_name_for_empty_messages(self):
        assert error_payload(RuntimeError())["error"]["message"] == "RuntimeError"

    def test_error_payload_from_string(self):
        assert error_payload("boom") == {
            "error": {"message": "boom", "type": "server_error", "code": None}
        }

    def test_error_payload_from_mapping(self):
        assert error_payload(
            {"message": "nope", "type": "invalid_request_error", "code": "bad_request"}
        ) == {
            "error": {"message": "nope", "type": "invalid_request_error", "code": "bad_request"}
        }

    def test_error_payload_from_already_shaped_mapping(self):
        assert error_payload(
            {"error": {"message": "nope", "type": "invalid_request_error", "code": None}}
        ) == {"error": {"message": "nope", "type": "invalid_request_error", "code": None}}

    def test_error_event_frame_round_trips(self):
        frame = sse_event(SSE_EVENT_ERROR, error_payload(ValueError("bad")))
        event, payload = _parse_frame(frame)
        assert event == SSE_EVENT_ERROR
        assert payload == {"error": {"message": "bad", "type": "ValueError", "code": None}}
