"""Cross-request Whisper window batching (Phase 10-2).

The suite pins the batcher's contract without model weights:

* window formation follows faster-whisper's own VAD/``collect_chunks`` path
  (a >60 s request forms multiple 30 s windows, each tagged with its offset);
* a batch carries one prompt per row, so requests with different detected
  languages share one ``encode``/``generate`` call;
* segment offsets are stitched back to original-audio time through
  ``restore_speech_timestamps``;
* a poisoned request/window errors alone while its neighbours complete;
* all three fallbacks (``len == 1``, flag off, batcher failure) use the
  per-request ``pipeline.transcribe`` path;
* scheduler admission counts true 30 s windows instead of one window per
  request;
* the batched payload has the same ``{"text", "segments", "info"}`` structure
  the sequential handler serialises.

faster-whisper's CT2 binding, Silero VAD and the feature extractor are stubbed;
the real ``collect_chunks``, ``pad_or_trim``, tokenizer wrapper,
``_split_segments_by_timestamps`` and ``restore_speech_timestamps`` are used, so
the test exercises the same plumbing the production path does.
"""

import dataclasses
import json
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
from faster_whisper.transcribe import Segment, TranscriptionInfo, TranscriptionOptions, WhisperModel

from harmonyspeech.common.artifacts import AudioArtifact
from harmonyspeech.common.config import DeviceConfig, ModelConfig
from harmonyspeech.common.inputs import SpeechTranscribeRequestInput
from harmonyspeech.common.request import EngineRequest, RequestStatus
from harmonyspeech.processing.scheduler import (
    LaneCredits,
    Scheduler,
    plan_lane_admission,
    whisper_window_count,
)
from harmonyspeech.task_handler import model_runner_base, whisper_batching
from harmonyspeech.task_handler.model_runner_base import ModelRunnerBase, WhisperRequestAbortedError
from harmonyspeech.task_handler.whisper_batching import (
    WHISPER_WINDOW_BATCH_SIZE,
    WhisperWindowBatcher,
    whisper_batched_inference_enabled,
)

SAMPLE_RATE = 16000

#: Fake token ids. The only invariant that matters is the upstream one:
#: text tokens < eot < timestamp tokens.
EOT = 1000
NO_TIMESTAMPS = 2000
TIMESTAMP_BEGIN = NO_TIMESTAMPS + 1
SOT = 1
SOT_PREV = 2
TRANSCRIBE = 3
TRANSLATE = 4
LANGUAGE_TOKENS = {"en": 10, "de": 11}


# ---------------------------------------------------------------------------
# Fake faster-whisper model (real wrappers, stubbed CT2 / VAD / features)
# ---------------------------------------------------------------------------


@dataclass
class _Result:
    sequences_ids: list
    scores: list
    no_speech_prob: float


class _Encoded:
    def __init__(self, markers):
        self.markers = markers


class _FakeFeatureExtractor:
    sampling_rate = SAMPLE_RATE
    chunk_length = 30
    nb_max_frames = 3000
    time_per_frame = 0.01
    feature_size = 80
    mel_filters = np.zeros((80, 201), dtype=np.float32)

    def __call__(self, waveform):
        seconds = len(waveform) / float(self.sampling_rate)
        marker = int(round(seconds * 1000))
        frames = max(2, int(round(seconds / self.time_per_frame)))
        out = np.zeros((80, frames + 1), dtype=np.float32)
        out[0, 0] = float(marker)
        return out


class _FakeHFTokenizer:
    def __init__(self, language_tokens=None, token_text=None):
        self.language_tokens = language_tokens or LANGUAGE_TOKENS
        self.token_text = token_text if token_text is not None else {}
        self.special = {
            "<|startoftranscript|>": SOT,
            "<|startofprev|>": SOT_PREV,
            "<|transcribe|>": TRANSCRIBE,
            "<|translate|>": TRANSLATE,
            "<|notimestamps|>": NO_TIMESTAMPS,
            "<|nospeech|>": 5,
            "<|startoflm|>": 6,
            "<|endoftext|>": EOT,
        }

    def token_to_id(self, token):
        if token.startswith("<|") and token.endswith("|>"):
            body = token[2:-2]
            if body in self.language_tokens:
                return self.language_tokens[body]
        return self.special.get(token)

    def encode(self, text, add_special_tokens=False):
        return type("Encoded", (), {"ids": [1] if text else []})()

    def decode(self, tokens):
        return "".join(self.token_text.get(int(token), "") for token in tokens)


class _FakeCT2:
    is_multilingual = True
    n_mels = 80
    device = "cpu"

    def __init__(self, language_by_marker=None):
        self.language_by_marker = language_by_marker or {}
        self.generate_calls = []
        self.detect_calls = 0
        self._marker_token = {}
        self._token_text = {}
        self._next_token = 100

    def token_for_marker(self, marker):
        marker = int(marker)
        if marker not in self._marker_token:
            self._marker_token[marker] = self._next_token
            self._token_text[self._next_token] = f"r{marker}"
            self._next_token += 1
        return self._marker_token[marker]

    def detect_language(self, encoder_output):
        self.detect_calls += 1
        results = []
        for marker in encoder_output.markers:
            language = self.language_by_marker.get(int(marker), "en")
            results.append([(f"<|{language}|>", 0.99)])
        return results

    def generate(self, encoder_output, prompts, **kwargs):
        self.generate_calls.append(
            {"prompts": [list(prompt) for prompt in prompts], "kwargs": kwargs, "markers": list(encoder_output.markers)}
        )
        results = []
        for marker in encoder_output.markers:
            token = self.token_for_marker(marker)
            results.append(
                _Result(
                    sequences_ids=[[TIMESTAMP_BEGIN + 10, token, TIMESTAMP_BEGIN + 60]],
                    scores=[-0.5],
                    no_speech_prob=0.01,
                )
            )
        return results


class _FakeWhisperModel:
    """Enough of ``WhisperModel`` for the batcher, with real prompt/split helpers."""

    get_prompt = WhisperModel.get_prompt
    _split_segments_by_timestamps = WhisperModel._split_segments_by_timestamps

    def __init__(self, language_by_marker=None, multilingual=True):
        self.ct2 = _FakeCT2(language_by_marker=language_by_marker)
        self.model = self.ct2
        self.feature_extractor = _FakeFeatureExtractor()
        self.hf_tokenizer = _FakeHFTokenizer(token_text=self.ct2._token_text)
        self.frames_per_second = SAMPLE_RATE // 160
        self.time_precision = 0.02
        self.input_stride = 2
        self.max_length = 448
        self.ct2.is_multilingual = multilingual
        self.encode_calls = 0

    def encode(self, features):
        features = np.asarray(features)
        if features.ndim == 2:
            features = features[None, ...]
        self.encode_calls += 1
        return _Encoded(features[:, 0, 0])


class _DecodingTokenizer:
    """Wrapper that makes ``decode`` fail for one marker's text token."""

    def __init__(self, tokenizer, failing_text):
        self._tokenizer = tokenizer
        self._failing_text = failing_text

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def decode(self, tokens):
        text = self._tokenizer.decode(tokens)
        if text == self._failing_text:
            raise RuntimeError(f"decode poisoned for {text}")
        return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _whisper_config(name="whisper", *, max_batch_size=16, num_workers=None, batched_inference=None):
    kwargs = {}
    if num_workers is not None:
        kwargs["num_workers"] = num_workers
    if batched_inference is not None:
        kwargs["batched_inference"] = batched_inference
    return ModelConfig(
        name=name,
        model="dummy/whisper",
        model_type="FasterWhisper",
        max_batch_size=max_batch_size,
        device_config=DeviceConfig("cpu"),
        dtype="float32",
        **kwargs,
    )


def _audio(seconds):
    return AudioArtifact.from_numpy(np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32), SAMPLE_RATE)


def _request(
    request_id,
    seconds,
    *,
    model="whisper",
    request_cls=SpeechTranscribeRequestInput,
    task="transcribe",
    language=None,
    prompt=None,
):
    return EngineRequest(
        request_id=request_id,
        request_data=request_cls(
            request_id=request_id,
            requested_model=model,
            model=model,
            input_audio=None,
            task=task,
            language=language,
            prompt=prompt,
        ),
        arrival_time=0.0,
        artifacts={"input_audio": _audio(seconds)},
    )


def _clips_for(seconds):
    """VAD stand-in: split continuous speech into 30 s clips."""

    def fake_get_speech_timestamps(audio, vad_options=None, sampling_rate=SAMPLE_RATE, **kwargs):
        total = len(audio)
        clips = []
        position = 0
        while position < total:
            end = min(position + int(30 * SAMPLE_RATE), total)
            clips.append({"start": position, "end": end})
            position = end
        return clips

    return fake_get_speech_timestamps


def _fixed_clips(*spans):
    def fake_get_speech_timestamps(audio, vad_options=None, sampling_rate=SAMPLE_RATE, **kwargs):
        return [{"start": int(start * SAMPLE_RATE), "end": int(end * SAMPLE_RATE)} for start, end in spans]

    return fake_get_speech_timestamps


def _run_batcher(batcher, artifacts, requests, *, clips=None, suppress_tokens=(1, 2, 3)):
    with patch.object(whisper_batching, "get_speech_timestamps", clips or _clips_for(0)), patch.object(
        whisper_batching, "get_suppressed_tokens", lambda tokenizer, values: suppress_tokens
    ):
        return batcher.transcribe_batch(artifacts, requests)


# ---------------------------------------------------------------------------
# 1. Window formation from clip timestamps
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_multi_window_request_forms_three_windows_and_stitches_offsets():
    model = _FakeWhisperModel()
    batcher = WhisperWindowBatcher(model)
    request = _request("long", 65.0)

    pairs = _run_batcher(
        batcher, [request.artifacts["input_audio"]], [request], clips=_clips_for(65.0)
    )

    payload, error = pairs[0]
    assert error is None
    assert batcher.batches_formed == 1
    assert batcher.windows_batched == 3, "65 s of speech is three 30 s windows"
    assert payload["text"] == "r30000r30000r5000"
    assert len(payload["segments"]) == 3
    assert [segment["seek"] for segment in payload["segments"]] == [0, 3000, 6000]
    # The upstream single-segment branch starts at the window offset and ends at
    # the last predicted timestamp (60 * 0.02 = 1.2 s into the window).
    assert [segment["start"] for segment in payload["segments"]] == pytest.approx([0.0, 30.0, 60.0])
    assert [segment["end"] for segment in payload["segments"]] == pytest.approx([1.2, 31.2, 61.2])
    assert payload["info"]["duration"] == pytest.approx(65.0)
    assert payload["info"]["duration_after_vad"] == pytest.approx(65.0)


@pytest.mark.unit
def test_window_offsets_restore_original_audio_time_across_vad_gaps():
    model = _FakeWhisperModel()
    batcher = WhisperWindowBatcher(model)
    request = _request("gapped", 65.0)

    # 20 s speech, 5 s silence, 25 s speech, 5 s silence, 10 s speech.
    pairs = _run_batcher(
        batcher,
        [request.artifacts["input_audio"]],
        [request],
        clips=_fixed_clips((0, 20), (25, 50), (55, 65)),
    )

    payload, error = pairs[0]
    assert error is None
    assert batcher.windows_batched == 3
    # collect_chunks offsets live on the speech-only timeline (0, 20, 45);
    # restore_speech_timestamps adds the removed silence back (0, 5, 10).
    assert [segment["start"] for segment in payload["segments"]] == pytest.approx([0.0, 25.0, 55.0])
    assert [segment["end"] for segment in payload["segments"]] == pytest.approx([1.2, 26.2, 56.2])
    assert [segment["seek"] for segment in payload["segments"]] == [0, 2000, 4500]


@pytest.mark.unit
def test_window_batch_cap_splits_a_long_step_into_multiple_formed_batches():
    model = _FakeWhisperModel()
    batcher = WhisperWindowBatcher(model, window_batch_size=2)
    # 4 windows (120 s of continuous speech) must become two formed batches.
    request = _request("very-long", 120.0)

    pairs = _run_batcher(batcher, [request.artifacts["input_audio"]], [request], clips=_clips_for(120.0))

    payload, error = pairs[0]
    assert error is None
    assert batcher.batches_formed == 2
    assert batcher.windows_batched == 4
    assert len(model.ct2.generate_calls) == 2
    assert [len(call["markers"]) for call in model.ct2.generate_calls] == [2, 2]
    assert len(payload["segments"]) == 4
    assert WHISPER_WINDOW_BATCH_SIZE == 8


# ---------------------------------------------------------------------------
# 2. Per-request prompts / language detection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_one_batch_carries_a_prompt_per_request_language():
    model = _FakeWhisperModel(language_by_marker={4000: "de", 5000: "en"})
    batcher = WhisperWindowBatcher(model)
    requests = [_request("de", 4.0), _request("en", 5.0)]

    pairs = _run_batcher(
        batcher,
        [request.artifacts["input_audio"] for request in requests],
        requests,
        clips=_clips_for(5.0),
    )

    assert all(error is None for _payload, error in pairs)
    assert len(model.ct2.generate_calls) == 1, "one encode + one generate for the whole step"
    prompts = model.ct2.generate_calls[0]["prompts"]
    assert prompts == [
        [SOT, LANGUAGE_TOKENS["de"], TRANSCRIBE, NO_TIMESTAMPS],
        [SOT, LANGUAGE_TOKENS["en"], TRANSCRIBE, NO_TIMESTAMPS],
    ]
    # The batched encoder output is reused for detection: one CT2 detect call.
    assert model.ct2.detect_calls == 1
    assert [payload["info"]["language"] for payload, _error in pairs] == ["de", "en"]


@pytest.mark.unit
def test_english_only_model_skips_language_detection():
    model = _FakeWhisperModel(multilingual=False)
    batcher = WhisperWindowBatcher(model)
    request = _request("en-only", 4.0)

    pairs = _run_batcher(batcher, [request.artifacts["input_audio"]], [request], clips=_clips_for(4.0))

    payload, error = pairs[0]
    assert error is None
    assert model.ct2.detect_calls == 0
    assert payload["info"]["language"] == "en"
    assert payload["info"]["language_probability"] == 1
    assert payload["info"]["all_language_probs"] is None


# ---------------------------------------------------------------------------
# 2b. Task / language hint / initial prompt (Phase 10-4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_translate_task_uses_the_translate_prompt_token():
    model = _FakeWhisperModel()
    batcher = WhisperWindowBatcher(model)
    request = _request("translate", 0.2, task="translate")

    pairs = _run_batcher(batcher, [request.artifacts["input_audio"]], [request], clips=_clips_for(0.2))

    payload, error = pairs[0]
    assert error is None
    prompt = model.ct2.generate_calls[0]["prompts"][0]
    assert TRANSLATE in prompt
    assert TRANSCRIBE not in prompt
    assert payload["info"]["transcription_options"]["initial_prompt"] is None


@pytest.mark.unit
def test_mixed_task_batch_carries_one_prompt_per_row():
    model = _FakeWhisperModel()
    batcher = WhisperWindowBatcher(model)
    requests = [_request("asr", 0.2), _request("mt", 0.3, task="translate")]

    pairs = _run_batcher(
        batcher,
        [request.artifacts["input_audio"] for request in requests],
        requests,
        clips=_clips_for(1.0),
    )

    assert all(error is None for _payload, error in pairs)
    prompts = model.ct2.generate_calls[0]["prompts"]
    assert TRANSCRIBE in prompts[0] and TRANSLATE not in prompts[0]
    assert TRANSLATE in prompts[1] and TRANSCRIBE not in prompts[1]


@pytest.mark.unit
def test_language_hint_skips_detection_and_is_reported():
    model = _FakeWhisperModel()
    batcher = WhisperWindowBatcher(model)
    request = _request("hinted", 0.2, language="de")

    pairs = _run_batcher(batcher, [request.artifacts["input_audio"]], [request], clips=_clips_for(0.2))

    payload, error = pairs[0]
    assert error is None
    assert model.ct2.detect_calls == 0
    assert payload["info"]["language"] == "de"
    assert payload["info"]["language_probability"] == 1
    assert payload["info"]["all_language_probs"] is None


@pytest.mark.unit
def test_initial_prompt_travels_as_previous_tokens():
    model = _FakeWhisperModel()
    batcher = WhisperWindowBatcher(model)
    request = _request("prompted", 0.2, prompt="Harmony")

    pairs = _run_batcher(batcher, [request.artifacts["input_audio"]], [request], clips=_clips_for(0.2))

    payload, error = pairs[0]
    assert error is None
    prompt = model.ct2.generate_calls[0]["prompts"][0]
    # Upstream parity: get_prompt prepends <|startofprev|> for an initial prompt.
    assert SOT_PREV in prompt
    assert payload["info"]["transcription_options"]["initial_prompt"] == "Harmony"


# ---------------------------------------------------------------------------
# 3. Poisoned request / window isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_poisoned_request_collection_errors_alone():
    model = _FakeWhisperModel()
    original = model.feature_extractor

    class _PoisonedExtractor:
        sampling_rate = original.sampling_rate
        chunk_length = original.chunk_length

        def __call__(self, waveform):
            if abs(len(waveform) / float(SAMPLE_RATE) - 0.3) < 1e-9:
                raise RuntimeError("feature extraction exploded")
            return original(waveform)

    model.feature_extractor = _PoisonedExtractor()
    batcher = WhisperWindowBatcher(model)
    requests = [_request("ok-1", 0.2), _request("poison", 0.3), _request("ok-2", 0.4)]

    pairs = _run_batcher(
        batcher,
        [request.artifacts["input_audio"] for request in requests],
        requests,
        clips=_clips_for(1.0),
    )

    assert pairs[0][1] is None and pairs[0][0]["text"] == "r200"
    assert isinstance(pairs[1][1], RuntimeError)
    assert "exploded" in str(pairs[1][1])
    assert pairs[2][1] is None and pairs[2][0]["text"] == "r400"


@pytest.mark.unit
def test_poisoned_window_decode_errors_only_its_request():
    model = _FakeWhisperModel()
    batcher = WhisperWindowBatcher(model)
    requests = [_request("ok", 0.2), _request("poison", 0.3)]

    # Replace the second request's tokenizer after detection with a failing one.
    original_tokenizer_for = batcher._tokenizer_for

    def tokenizer_for(request_windows):
        tokenizer = original_tokenizer_for(request_windows)
        if request_windows.request_id == "poison":
            wrapped = _DecodingTokenizer(tokenizer, "r300")
            request_windows.tokenizer = wrapped
            return wrapped
        return tokenizer

    batcher._tokenizer_for = tokenizer_for

    pairs = _run_batcher(
        batcher,
        [request.artifacts["input_audio"] for request in requests],
        requests,
        clips=_clips_for(1.0),
    )

    assert pairs[0][1] is None and pairs[0][0]["text"] == "r200"
    assert isinstance(pairs[1][1], RuntimeError)
    assert "decode poisoned" in str(pairs[1][1])


@pytest.mark.unit
def test_aborted_request_errors_alone_without_generating_its_windows():
    model = _FakeWhisperModel()
    batcher = WhisperWindowBatcher(model)
    requests = [_request("ok", 0.2), _request("aborted", 0.3)]
    requests[1].status = RequestStatus.FINISHED_ABORTED

    pairs = _run_batcher(
        batcher,
        [request.artifacts["input_audio"] for request in requests],
        requests,
        clips=_clips_for(1.0),
    )

    assert pairs[0][1] is None
    assert isinstance(pairs[1][1], WhisperRequestAbortedError)
    assert len(model.ct2.generate_calls) == 1
    assert model.ct2.generate_calls[0]["markers"] == [200.0]


# ---------------------------------------------------------------------------
# 4. Payload shape parity with the sequential path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_batched_payload_matches_sequential_payload_structure():
    model = _FakeWhisperModel()
    batcher = WhisperWindowBatcher(model)
    request = _request("shape", 0.2)

    pairs = _run_batcher(batcher, [request.artifacts["input_audio"]], [request], clips=_clips_for(0.2))
    payload, error = pairs[0]

    assert error is None
    assert set(payload) == {"text", "segments", "info"}
    assert set(payload["segments"][0]) == {field.name for field in dataclasses.fields(Segment)}
    assert set(payload["info"]) == {field.name for field in dataclasses.fields(TranscriptionInfo)}
    assert set(payload["info"]["transcription_options"]) == {
        field.name for field in dataclasses.fields(TranscriptionOptions)
    }
    segment = payload["segments"][0]
    assert segment["words"] is None
    assert segment["temperature"] == 0.0
    assert segment["tokens"] == [TIMESTAMP_BEGIN + 10, 100, TIMESTAMP_BEGIN + 60]
    assert payload["info"]["transcription_options"]["without_timestamps"] is True
    assert payload["info"]["transcription_options"]["condition_on_previous_text"] is False
    assert payload["info"]["transcription_options"]["temperatures"] == [0.0]
    assert payload["info"]["transcription_options"]["clip_timestamps"] == [
        {"start": 0, "end": int(0.2 * SAMPLE_RATE)}
    ]
    # The dict is JSON-serialisable with exactly the keys the handler emits.
    assert json.loads(json.dumps(payload))["text"] == "r200"


# ---------------------------------------------------------------------------
# 5. Handler wiring + all three fallbacks
# ---------------------------------------------------------------------------


@dataclass
class _SequentialSegment:
    text: str


@dataclass
class _SequentialInfo:
    language: str = "en"
    language_probability: float = 0.9
    duration: float = 0.1


class _Pipeline:
    calls = []

    def __init__(self, model=None):
        self.model = model
        self.last_speech_timestamp = 0.0
        type(self).calls.append(self)

    def transcribe(self, audio, batch_size=16, **kwargs):
        self.last_speech_timestamp = 0.0
        return [_SequentialSegment("sequential")], _SequentialInfo()


def _runner(model, config):
    runner = ModelRunnerBase.__new__(ModelRunnerBase)
    runner.model = model
    runner.device = "cpu"
    runner.model_config = config
    runner._baseline_state = None
    return runner


@pytest.mark.unit
def test_handler_uses_batched_path_for_multiple_requests():
    _Pipeline.calls = []
    config = _whisper_config(batched_inference=True)
    runner = _runner(_FakeWhisperModel(), config)
    requests = [_request("a", 0.2), _request("b", 0.3)]
    inputs = [request.artifacts["input_audio"] for request in requests]

    with patch.object(whisper_batching, "get_speech_timestamps", _clips_for(1.0)), patch.object(
        whisper_batching, "get_suppressed_tokens", lambda tokenizer, values: (1,)
    ), patch.object(model_runner_base, "BatchedInferencePipeline", _Pipeline):
        outputs = runner._execute_faster_whisper(inputs, requests)

    assert [output.request_id for output in outputs] == ["a", "b"]
    assert [json.loads(output.result_data.output)["text"] for output in outputs] == ["r200", "r300"]
    assert _Pipeline.calls == [], "the batched path must not build a pipeline"


@pytest.mark.unit
def test_handler_len_one_takes_the_sequential_pipeline_path():
    _Pipeline.calls = []
    config = _whisper_config(batched_inference=True)
    runner = _runner(_FakeWhisperModel(), config)
    request = _request("solo", 0.2)

    with patch.object(model_runner_base, "WhisperWindowBatcher") as batcher_cls, patch.object(
        model_runner_base, "BatchedInferencePipeline", _Pipeline
    ):
        batcher_cls.side_effect = AssertionError("batcher must not run for one request")
        outputs = runner._execute_faster_whisper([request.artifacts["input_audio"]], [request])

    assert json.loads(outputs[0].result_data.output)["text"] == "sequential"


@pytest.mark.unit
def test_handler_flag_off_always_takes_the_sequential_path():
    _Pipeline.calls = []
    config = _whisper_config(batched_inference=False)
    runner = _runner(_FakeWhisperModel(), config)
    requests = [_request("a", 0.2), _request("b", 0.3)]
    inputs = [request.artifacts["input_audio"] for request in requests]

    with patch.object(model_runner_base, "WhisperWindowBatcher") as batcher_cls, patch.object(
        model_runner_base, "BatchedInferencePipeline", _Pipeline
    ):
        batcher_cls.side_effect = AssertionError("batcher must not run when the flag is off")
        outputs = runner._execute_faster_whisper(inputs, requests)

    assert [json.loads(output.result_data.output)["text"] for output in outputs] == [
        "sequential",
        "sequential",
    ]


@pytest.mark.unit
def test_handler_batcher_failure_falls_back_to_per_request_pipeline():
    _Pipeline.calls = []
    config = _whisper_config(batched_inference=True)
    runner = _runner(_FakeWhisperModel(), config)
    requests = [_request("a", 0.2), _request("b", 0.3)]
    inputs = [request.artifacts["input_audio"] for request in requests]

    class _ExplodingBatcher:
        def __init__(self, model=None):
            self.model = model

        def transcribe_batch(self, audio_artifacts, requests_to_batch):
            raise RuntimeError("CT2 exploded")

    with patch.object(model_runner_base, "WhisperWindowBatcher", _ExplodingBatcher), patch.object(
        model_runner_base, "BatchedInferencePipeline", _Pipeline
    ):
        outputs = runner._execute_faster_whisper(inputs, requests)

    assert [json.loads(output.result_data.output)["text"] for output in outputs] == [
        "sequential",
        "sequential",
    ]
    assert runner._whisper_batch_fallbacks == 1
    assert runner.whisper_lane_stats()["batch_fallbacks"] == 1
    from harmonyspeech.engine.whisper_lane_metrics import render_whisper_lane_metrics

    text = render_whisper_lane_metrics({"enabled": True, "models": {"whisper": runner.whisper_lane_stats()}})
    assert 'hse_whisper_batch_fallbacks_total{model="whisper"} 1' in text


@pytest.mark.unit
def test_batched_inference_flag_resolution():
    # Opt-in for Whisper: unset keeps the pre-10-2 sequential path.
    assert whisper_batched_inference_enabled(_whisper_config()) is False
    assert whisper_batched_inference_enabled(_whisper_config(batched_inference=True)) is True
    assert whisper_batched_inference_enabled(_whisper_config(batched_inference=False)) is False


# ---------------------------------------------------------------------------
# 6. Admission now counts true 30 s windows
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "seconds,expected",
    [(0.1, 1), (30.0, 1), (30.01, 2), (60.0, 2), (65.0, 3), (300.0, 10)],
)
def test_whisper_window_count(seconds, expected):
    assert whisper_window_count(_request("r", seconds)) == expected


@pytest.mark.unit
def test_whisper_window_count_unknown_duration_costs_one():
    request = EngineRequest(
        request_id="r",
        request_data=SpeechTranscribeRequestInput(request_id="r", requested_model="whisper", model="whisper"),
        arrival_time=0.0,
    )
    assert whisper_window_count(request) == 1


@pytest.mark.unit
def test_scheduler_admits_by_window_budget_not_request_count():
    config = _whisper_config(max_batch_size=16)
    scheduler = Scheduler(
        [config],
        whisper_admission=lambda active_windows=0: {"enabled": True, "allowed_windows": 6},
    )
    for index in range(3):
        scheduler.add_request(_request(f"long-{index}", 65.0))

    scheduled = scheduler.schedule().scheduled_requests_per_model["whisper"]

    # Each 65 s request costs 3 windows; a 6-window allowance admits exactly 2.
    assert len(scheduled) == 2
    assert len(scheduler.waiting) == 1


@pytest.mark.unit
def test_scheduler_window_budget_subtracts_running_windows():
    config = _whisper_config(max_batch_size=16)
    scheduler = Scheduler(
        [config],
        whisper_admission=lambda active_windows=0: {"enabled": True, "allowed_windows": 4},
    )
    running = _request("running", 65.0)
    running.set_running()
    scheduler.running.append(running)
    scheduler.add_request(_request("waiting", 30.0))

    scheduled = scheduler.schedule().scheduled_requests_per_model.get("whisper", [])

    # 3 windows are in flight; the 1-window request fits the remaining headroom.
    assert [request.request_id for request in scheduled] == ["waiting"]


@pytest.mark.unit
def test_scheduler_window_budget_progress_floor_for_a_single_long_request():
    config = _whisper_config(max_batch_size=16)
    scheduler = Scheduler(
        [config],
        whisper_admission=lambda active_windows=0: {"enabled": True, "allowed_windows": 1},
    )
    scheduler.add_request(_request("only-long", 65.0))

    scheduled = scheduler.schedule().scheduled_requests_per_model["whisper"]

    # A 3-window request against a 1-window allowance would starve forever; the
    # floor admits it when nothing is in flight (Phase 10-1 liveness preserved).
    assert [request.request_id for request in scheduled] == ["only-long"]


@pytest.mark.unit
def test_scheduler_window_budget_respects_a_hard_stop():
    config = _whisper_config(max_batch_size=16)
    scheduler = Scheduler(
        [config],
        whisper_admission=lambda active_windows=0: {"enabled": True, "allowed_windows": 0},
    )
    scheduler.add_request(_request("waiting", 5.0))

    outputs = scheduler.schedule()

    assert outputs.scheduled_requests_per_model == {}
    assert [request.request_id for request in scheduler.waiting] == ["waiting"]


@pytest.mark.unit
def test_plan_lane_admission_skips_an_oversized_request_but_admits_a_smaller_one():
    long_request = _request("long", 65.0)
    short_request = _request("short", 5.0)
    selected, leftover = plan_lane_admission(
        [(long_request, "long"), (short_request, "short")],
        capacity=2,
        credits=LaneCredits(),
        window_costs={id(long_request): 3, id(short_request): 1},
        window_budget=1,
    )

    assert [request.request_id for request in selected] == ["short"]
    assert [request.request_id for request in leftover] == ["long"]
