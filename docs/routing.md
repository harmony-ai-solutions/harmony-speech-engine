# Request Routing Mechanism

Harmony Speech Engine aims at providing a hollistic approach at hosting different Neural Network models
which provide functionality for AI Speech related tasks.

A lot of current AI Speech Frameworks consist of multiple models or weight elements, which perform different
tasks or steps in the speech processing of the framework. Harmony Speech Engine allows for hosting and scaling the
individual models independently, and requests to a specific framework will be routed between the individual models
as required.

Also, tasks like embedding a speaker or transcribing an audio file are not always required as part of the processing,
and calling them repeatedly causes unnecessary compute time and might slow down an inference task. Therefore,
Harmony Speech Engine provides funtionality in the routing to omit redundant steps, as well as the option of
calling a specific model directly, e.g. if you already have a speaker embedding or transcription metadata.

## Declarative pipelines (the extension point)

Routing is **declarative**. Every model toolchain (family) is described once, in
`harmonyspeech/processing/pipeline_registry.py`, as a `PipelineSpec`:

| Piece | Meaning |
|---|---|
| `steps` | Ordered `Step`s. Each step names the `model_type`s that can execute it (in preference order), the request fields/artifacts it needs (`needs`), the request fields its typed output fills (`produces`) and the fields its output supersedes (`clears`). |
| `entry_rules` | Ordered `EntryRule`s that map an incoming request shape onto a starting step (and optionally an early `stop_after`). |
| `final_step` | The step whose output is the request's response. |
| `capabilities` | What the pipeline can serve: `tts`, `embed`, `vc`, `stt`, `translation`, `vad`, `restore`. Drives `/v1/models`. |

Two functions consume that declaration:

* **Routing** — `next_step(spec, state, request, model_configs)`: resolves the concrete
  model for the request's current step. The first `requires` entry backed by a
  configured model wins; language-variant steps additionally require the
  configured `language` to match the request. When nothing matches, the request
  fails immediately with an error naming the pipeline and the step (it is never
  queued to wait forever).
* **Forwarding** — `advance(spec, engine_request, result)`: stores the finished
  step's typed artifact under the request field declared in `produces`,
  increments the step index and reports whether the pipeline is finished. No
  code inspects output classes, so a step can only ever feed the field it
  declared.

A request's position lives on `EngineRequest.pipeline_state` (`spec_id`,
`step_index`, `stop_after`). It is resolved once at admission from the entry
rules and is owned by the state machine afterwards — a forwarded request never
re-resolves its entry point, which is what made the old field-presence routing
fragile.

### Adding a new toolchain

1. Add one `PipelineSpec` to `PIPELINE_SPECS` (steps, entry rules, capabilities).
2. Implement the executors' handlers for any new `model_type` (input preparation
   and execution dispatch), as for any other model.

Nothing else changes: routing, forwarding, startup validation and `/v1/models`
all read the registry. `tests/unit/inference_flow/test_pipeline_routing.py`
contains a worked example that registers a two-step toy toolchain in a test and
runs it end to end.

### Single-step executors and direct model access

A request may name a concrete configured model instead of a toolchain
(`"model": "kitten-tts-mini"`, `"model": "ov1-synthesizer-en"`). Such requests
skip the pipeline machinery and are executed by that one executor. Toolchain
ids (`openvoice_v2`, `chatterbox`, ...) route through the pipeline; the registry
also declares single-step pipelines (`kitten_tts`, `fasterwhisper`,
`silero_vad`) so a request may address those by family id too.

## VAD backend selection (`input_vad_mode`)

The OpenVoice pipelines need a VAD/ASR step to segment the reference audio. The
step declares a backend preference list (`FasterWhisper`, then `SileroVAD`), and
`input_vad_mode` reorders it per request:

| `input_vad_mode` | Behaviour |
|---|---|
| `"whisper"` / omitted | Faster-Whisper (the default and historical backend). |
| `"silero"` | Silero VAD when one is configured; otherwise Faster-Whisper with a warning in the log. |

Both backends emit the same `VadArtifact` segment schema (`start`/`end` in
seconds plus `text`), so the encoder consumes either transparently. Silero
segments carry no transcript, which the encoder's text-length filter
accommodates.

## Entry points

| Request shape | Starts at | Stops after |
|---|---|---|
| Embedding request with audio, no VAD data | VAD | embed |
| Embedding request with VAD data (or no audio) | embed | embed |
| Cloning TTS with audio, no VAD data | VAD | final step |
| Cloning TTS with audio + VAD data | embed | final step |
| Cloning TTS with a precomputed embedding | synthesize | final step |
| Single-speaker TTS (`mode: "single_speaker_tts"`) | synthesize | synthesize |
| Voice conversion | tone conversion / convert | same step |
| Transcription / VAD | single VAD/ASR step | same step |

`mode: "single_speaker_tts"` is therefore a real entry point: it starts at the
synthesis step and finishes there, without mutating request fields. Chatterbox
with reference audio starts at its embedding step only when a dedicated
embedding executor is configured; otherwise the TTS executor embeds inline and
the request starts at synthesis.
