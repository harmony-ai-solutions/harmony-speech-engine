# Codebase Concerns

**Analysis Date:** 2026-02-28

## Tech Debt

**Missing Batched Inference:**
- Issue: Most model execution functions iterate over requests and perform inference one by one, missing out on GPU batching performance gains.
- Files: `harmonyspeech/task_handler/model_runner_base.py`
- Impact: Significantly reduced throughput and higher latency for concurrent requests.
- Fix approach: Refactor `_execute_*` methods to use batched versions of model inference calls.

**Hardcoded Routing Logic:**
- Issue: Complex routing for multi-step processing (e.g., OpenVoice, Harmonyspeech) is hardcoded into the engine using string-based `model_type` comparisons.
- Files: `harmonyspeech/engine/harmonyspeech_engine.py`
- Impact: High maintenance overhead when adding or modifying models/workflows.
- Fix approach: Implement a more flexible, configuration-driven routing or pipeline management system.

**Unoptimized MeloTTS Integration:**
- Issue: Integration of MeloTTS includes "super complicated, imperformant and underoptimized code" (according to code comments), including redundant model allocations.
- Files: `harmonyspeech/modeling/models/melo/inputs.py`, `harmonyspeech/modeling/models/melo/melo.py`
- Impact: Unnecessary memory usage and higher latency for MeloTTS-based generation.
- Fix approach: Rewrite the text normalization and Bert-based embedding logic to be more efficient.

**Generic Error Handling:**
- Issue: Widespread use of `ValueError` and generic exceptions instead of specialized error classes.
- Files: `harmonyspeech/endpoints/openai/serving_*`, `harmonyspeech/engine/async_harmonyspeech.py`
- Impact: Difficult to distinguish between different types of failures (validation, engine error, model error).
- Fix approach: Define a comprehensive set of domain-specific exception classes.

## Security Considerations

**API Key Fail-Open Behavior:**
- Issue: API key checks are disabled if `ENDPOINT_URL` is not set or empty, potentially exposing the service without authentication.
- Files: `auth/apikeys.py:108`
- Risk: Unauthorized access if the environment variable is misconfigured.
- Current mitigation: Relies on correct environment configuration.
- Recommendations: Implement a safer default behavior (fail-closed) or a mandatory configuration check.

**Non-Distributed API Key Cache:**
- Issue: `_api_key_cache` is a simple in-memory dictionary.
- Files: `auth/apikeys.py:31`
- Risk: In a distributed Ray environment, rate limits and API key validity may not be synchronized across worker actors.
- Recommendations: Use a distributed cache (like Redis) or a centralized auth service.

## Performance Bottlenecks

**Potential GIL Bottleneck:**
- Problem: The engine uses `ThreadPoolExecutor` to run executors in parallel.
- Files: `harmonyspeech/engine/harmonyspeech_engine.py:592`
- Cause: If the executor's work is not effectively releasing the GIL (e.g., heavy Python-based pre/post-processing), parallelism will be limited.
- Improvement path: Verify GIL release patterns in pre/post-processing; consider using `ProcessPoolExecutor` or Ray tasks for CPU-bound work.

**Redundant Text Normalization Allocations:**
- Problem: Bert is allocated for each normalized part of the text separately during MeloTTS inference.
- Files: `harmonyspeech/modeling/models/melo/inputs.py:30`
- Cause: Unoptimized pre-processing pipeline in inherited code.
- Improvement path: Cache or reuse the Bert model instance for the duration of the request or worker lifecycle.

## Fragile Areas

**Multi-Step Request Forwarding:**
- Files: `harmonyspeech/engine/harmonyspeech_engine.py:314` (`check_forward_processing`)
- Why fragile: Re-adding requests to the scheduler to achieve multi-step processing (e.g., VAD -> Embedding -> Synthesis) creates complex state management and tracking requirements.
- Safe modification: Carefully test any changes to the `step()` loop or `Scheduler` state transitions.

**String-Based Model Type Identification:**
- Files: `harmonyspeech/task_handler/model_runner_base.py:62`, `harmonyspeech/engine/harmonyspeech_engine.py`
- Why fragile: Reliance on magic strings for `model_type` makes it easy to introduce typos or break routing during refactoring.
- Safe modification: Use a central enumeration or constant-based mapping for model types.

## Scaling Limits

**Single-Instance Scheduler:**
- Current capacity: Single engine instance manages all requests.
- Limit: The centralized scheduler may become a bottleneck for extremely high request volumes or across many Ray workers.
- Scaling path: Distribute scheduling or use more efficient queueing mechanisms if Ray overhead becomes an issue.

## Missing Critical Features

**Streaming Support:**
- Problem: "Stream output is not yet supported" in TTS endpoints.
- Files: `harmonyspeech/endpoints/openai/serving_text_to_speech.py:68`
- Blocks: Real-time application use cases that require low time-to-first-byte.

**Silero VAD Routing:**
- Problem: "TODO: add switch for silero here using request.input_vad_mode"
- Files: `harmonyspeech/engine/harmonyspeech_engine.py:165`
- Blocks: Users from choosing alternative VAD engines via API parameters.

## Test Coverage Gaps

**Integration Testing:**
- What's not tested: Complex multi-step routing (VAD -> Embedding -> TTS) through the engine.
- Files: `harmonyspeech/engine/harmonyspeech_engine.py`
- Risk: Regressions in routing logic could break whole model workflows unnoticed.
- Priority: High
