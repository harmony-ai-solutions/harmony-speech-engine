# Phase 2-4: VllmOmniExecutor — Execute Model and Audio Extraction

## Objective

Implement `execute_model()` inside `VllmOmniExecutor`. This is the method called by `HarmonySpeechEngine.step()` with a batch of `EngineRequest` objects. It must:
1. Build the vllm-omni prompt for each request
2. Call `self.omni.generate([prompt])` for each request
3. Extract the audio tensor from `OmniRequestOutput.multimodal_output`
4. Encode audio as WAV base64
5. Return a list of `ExecutorResult` objects wrapping `TextToSpeechRequestOutput`

## Background: Audio Output Structure (confirmed from vllm-omni codebase)

From `vllm_omni/outputs.py` — `OmniRequestOutput.multimodal_output` property:
```python
@property
def multimodal_output(self) -> dict[str, Any]:
    # For pipeline outputs (TTS), navigates into request_output.outputs[0].multimodal_output
    for req_out in [self.request_output]:
        for output in getattr(req_out, "outputs", []):
            if mm := getattr(output, "multimodal_output", None):
                return mm
    return self._multimodal_output
```

The returned dict has structure:
```python
{
    "audio": torch.Tensor | List[torch.Tensor],  # audio waveform (1D or chunked list)
    "sr": int | List[int],                        # sample rate (or list if chunked)
}
```

Audio extraction pattern (from `serving_speech.py` and `end2end.py`):
```python
mm = stage_output.multimodal_output  # {"audio": ..., "sr": ...}

# Sample rate
sr_raw = mm.get("sr", 24000)
sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
sample_rate = int(sr_val.item() if hasattr(sr_val, "item") else sr_val)

# Audio data
audio_data = mm["audio"]
if isinstance(audio_data, list):
    import torch
    audio_tensor = torch.cat(audio_data, dim=-1)
else:
    audio_tensor = audio_data

audio_np = audio_tensor.float().detach().cpu().numpy().flatten()
```

## Batching Strategy

`execute_model` receives `requests_to_batch: List[EngineRequest]`. For vllm-omni, each request is dispatched **individually** to `omni.generate([single_prompt])`:

- **Why not batch?** `omni.generate([p1, p2])` batches at the vllm-omni level, but HSE's scheduler already handles batching and the `max_batch_size` in the config limits concurrent requests per call. Calling individually keeps error handling clean and avoids complexity from partial batch failures.
- **Performance**: For LLM-based TTS, throughput is dominated by the autoregressive generation, not the dispatch overhead.

Future optimization: could collect all prompts and call `omni.generate(all_prompts)` for true batching. This requires careful handling of multiple outputs and error isolation.

## File to Modify

**`harmonyspeech/executor/vllm_omni_executor.py`** — replace the `execute_model` stub

## Implementation

```python
def execute_model(
    self,
    requests_to_batch: List[EngineRequest],
) -> List[ExecutorResult]:
    """Execute TTS inference for a batch of requests using vllm-omni.

    Each request is processed individually (single-item list to omni.generate).
    Failed requests produce an ExecutorResult with output=None and finish_reason="error".

    Args:
        requests_to_batch: List of EngineRequest objects from the HSE scheduler.
                           Each request_data is expected to be TextToSpeechRequestInput.

    Returns:
        List of ExecutorResult, one per input request, in the same order.
    """
    import soundfile as sf  # noqa: PLC0415
    from harmonyspeech.common.outputs import TextToSpeechRequestOutput  # noqa: PLC0415

    outputs: List[ExecutorResult] = []

    for req in requests_to_batch:
        request_data = req.request_data
        req_id = req.request_id

        try:
            # 1. Build prompt dict from request data
            prompt = self._build_prompt(request_data)

            # 2. Run vllm-omni generation (blocking synchronous call)
            #    Returns List[OmniRequestOutput] — one entry for our single prompt
            omni_outputs = self.omni.generate([prompt])

            if not omni_outputs:
                raise RuntimeError(
                    f"omni.generate() returned empty output for request {req_id}"
                )

            # 3. Extract multimodal_output dict from the first (and only) OmniRequestOutput
            stage_output = omni_outputs[0]
            mm = stage_output.multimodal_output

            if mm is None:
                raise RuntimeError(
                    f"multimodal_output is None for request {req_id} — "
                    "the model did not produce audio output."
                )

            # 4. Extract sample rate
            sr_raw = mm.get("sr", 24000)
            sr_val = sr_raw[-1] if isinstance(sr_raw, list) and sr_raw else sr_raw
            sample_rate = int(sr_val.item() if hasattr(sr_val, "item") else sr_val)

            # 5. Extract and concatenate audio tensor
            audio_data = mm.get("audio")
            if audio_data is None:
                raise RuntimeError(
                    f"No 'audio' key in multimodal_output for request {req_id}. "
                    f"Available keys: {list(mm.keys())}"
                )

            if isinstance(audio_data, list):
                audio_tensor = torch.cat(audio_data, dim=-1)
            else:
                audio_tensor = audio_data

            # Ensure 1D float32 numpy array
            audio_np = audio_tensor.float().detach().cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            audio_np = audio_np.flatten()

            # 6. Encode as WAV base64
            buf = io.BytesIO()
            sf.write(buf, audio_np, samplerate=sample_rate, format="WAV")
            encoded_audio = base64.b64encode(buf.getvalue()).decode("utf-8")
            buf.close()

            # 7. Update request metrics and build result
            req.metrics.finished_time = time.time()
            result_data = TextToSpeechRequestOutput(
                request_id=req_id,
                text=getattr(request_data, "input_text", ""),
                output=encoded_audio,
                finish_reason="stop",
                metrics=req.metrics,
            )
            logger.debug(
                f"[VllmOmniExecutor] Request {req_id} completed: "
                f"{len(audio_np)} samples @ {sample_rate} Hz"
            )

        except Exception as e:
            # Log the error but don't crash — return error result for this request
            logger.error(
                f"[VllmOmniExecutor] Request {req_id} failed: {type(e).__name__}: {e}",
                exc_info=True,
            )
            req.metrics.finished_time = time.time()
            result_data = TextToSpeechRequestOutput(
                request_id=req_id,
                text=getattr(request_data, "input_text", ""),
                output=None,
                finish_reason="error",
                metrics=req.metrics,
            )

        outputs.append(ExecutorResult(
            request_id=req_id,
            input_data=request_data,
            result_data=result_data,
        ))

    return outputs
```

## Imports Checklist

The following are used and must be available at the top of the file (already included in Phase 2-1):
- `import base64` ✓
- `import io` ✓
- `import time` ✓
- `import torch` ✓
- `from loguru import logger` ✓
- `from harmonyspeech.common.request import EngineRequest, ExecutorResult` ✓

Lazy imported inside the function:
- `import soundfile as sf` — already in `requirements-common.txt`
- `from harmonyspeech.common.outputs import TextToSpeechRequestOutput`

## Notes on `TextToSpeechRequestOutput`

`TextToSpeechRequestOutput` (from `harmonyspeech/common/outputs.py`) requires:
- `request_id: str`
- `text: str` (the input text, not the output)
- `output: Optional[str]` (base64-encoded WAV bytes, or `None` on error)
- `finish_reason: str` ("stop" or "error")
- `metrics: RequestMetrics`

On error, `output=None` is returned. The engine's `check_forward_processing` will handle this gracefully since it checks `finish_reason`.

## Progress Checklist

- [ ] Replace `execute_model()` stub with full implementation
- [ ] Verify audio extraction: `mm["audio"]` as single tensor → correct 1D numpy array
- [ ] Verify audio extraction: `mm["audio"]` as list of tensors → concatenated correctly
- [ ] Verify sample rate extraction: single int and list both handled
- [ ] Verify error handling: exception in one request doesn't prevent other requests from processing
- [ ] Verify WAV encoding: `soundfile.write()` produces valid WAV bytes
- [ ] Verify base64 encoding: output is a valid UTF-8 string
- [ ] Verify `TextToSpeechRequestOutput` constructor call matches the actual class signature
  (check `harmonyspeech/common/outputs.py` — the `text` and `output` parameter names)
