"""Wire-contract tests for the Phase 12-1 long-form request fields.

The fields are accepted now and are inert until the serving layer wires them
(12-2); these tests pin the defaults so existing requests stay byte-identical.
"""

from harmonyspeech.common.inputs import TextToSpeechGenerationOptions, TextToSpeechRequestInput
from harmonyspeech.endpoints.openai.protocol import GenerationOptions, TextToSpeechRequest


def test_generation_options_long_form_defaults():
    opts = GenerationOptions()
    assert opts.split_text is False
    assert opts.chunk_size is None
    assert opts.crossfade_ms is None
    assert opts.sentence_pause_ms is None


def test_generation_options_accepts_long_form_fields():
    opts = GenerationOptions(split_text=True, chunk_size=240, crossfade_ms=30, sentence_pause_ms=120)
    assert opts.split_text is True
    assert opts.chunk_size == 240
    assert opts.crossfade_ms == 30
    assert opts.sentence_pause_ms == 120


def test_legacy_request_without_generation_options_is_unchanged():
    request = TextToSpeechRequest.model_validate({"model": "tts", "input": "hello"})
    assert request.generation_options is None


def test_request_input_passes_long_form_options_through():
    options = GenerationOptions(split_text=True, chunk_size=180, crossfade_ms=25, sentence_pause_ms=90)
    request = TextToSpeechRequest(model="tts", input="hello", generation_options=options)
    request_input = TextToSpeechRequestInput.from_openai("tts-1", request)
    assert request_input.generation_options is options
    assert request_input.generation_options.split_text is True
    assert request_input.generation_options.chunk_size == 180
    assert request_input.generation_options.crossfade_ms == 25
    assert request_input.generation_options.sentence_pause_ms == 90


def test_internal_generation_options_dataclass_defaults():
    opts = TextToSpeechGenerationOptions(seed=None, style=None, speed=None, pitch=None, energy=None)
    assert opts.split_text is False
    assert opts.chunk_size is None
    assert opts.crossfade_ms is None
    assert opts.sentence_pause_ms is None


def test_internal_watermark_suppression_is_not_a_wire_field():
    """Phase 12-2's per-chunk suppression flag is internal only."""
    from harmonyspeech.common.inputs import SpeechEmbeddingRequestInput

    options = GenerationOptions(split_text=True, chunk_size=120)
    request = TextToSpeechRequest(model="tts", input="hello", generation_options=options)
    request_input = TextToSpeechRequestInput.from_openai("tts-2", request)
    assert request_input.suppress_watermark is False
    assert "suppress_watermark" not in GenerationOptions.model_fields
    # The one-time reference-embedding sub-request may carry the originating
    # request's options (internal, not serialized).
    embed = SpeechEmbeddingRequestInput(
        request_id="embed-1",
        requested_model="tts",
        model="tts",
        generation_options=options,
    )
    assert embed.generation_options is options


def test_internal_request_timeout_override_is_not_a_wire_field():
    """Phase 12-2 follow-up: the per-chunk deadline override is internal only."""
    options = GenerationOptions(split_text=True, chunk_size=120)
    request = TextToSpeechRequest(model="tts", input="hello", generation_options=options)
    request_input = TextToSpeechRequestInput.from_openai("tts-timeout", request)
    # Wire requests never set it; None means "inherit the engine-wide timeout".
    assert request_input.request_timeout_s is None
    assert "request_timeout_s" not in TextToSpeechRequest.model_fields
    assert "request_timeout_s" not in GenerationOptions.model_fields


def test_chunk_request_timeout_override_formula():
    """Every chunk gets ``base_timeout * (chunk_count + 1)``; off stays off."""
    from types import SimpleNamespace

    from harmonyspeech.endpoints.openai.serving_text_to_speech import (
        OpenAIServingTextToSpeech,
    )

    serving = OpenAIServingTextToSpeech.__new__(OpenAIServingTextToSpeech)
    serving.engine = SimpleNamespace(engine=SimpleNamespace(request_timeout_s=120.0))
    assert serving._chunk_request_timeout_s(4) == 600.0
    # A disabled engine-wide timeout stays disabled (None = inherit).
    serving.engine = SimpleNamespace(engine=SimpleNamespace(request_timeout_s=0))
    assert serving._chunk_request_timeout_s(4) is None


def test_chunking_defaults_keep_single_shot_requests_unchunked():
    """Default-off fields must not enable the serving chunker."""
    from harmonyspeech.endpoints.openai.serving_text_to_speech import DEFAULT_CHUNK_SIZE

    options = GenerationOptions()
    assert options.split_text is False
    assert options.chunk_size is None
    assert DEFAULT_CHUNK_SIZE > 0


def test_chunked_join_defaults_are_tuned_but_explicit_values_win():
    """Phase 12-5: unset join timing gets 40/120 ms; explicit values win."""
    from harmonyspeech.endpoints.openai.serving_text_to_speech import (
        DEFAULT_CHUNK_CROSSFADE_MS,
        DEFAULT_CHUNK_SENTENCE_PAUSE_MS,
        _resolve_join_timing,
    )

    assert DEFAULT_CHUNK_CROSSFADE_MS == 40
    assert DEFAULT_CHUNK_SENTENCE_PAUSE_MS == 120

    # Wire defaults stay None; the tuning is applied only at the chunked path.
    assert GenerationOptions().crossfade_ms is None
    assert GenerationOptions().sentence_pause_ms is None
    assert _resolve_join_timing(None) == (40, 120)
    assert _resolve_join_timing(GenerationOptions()) == (40, 120)

    # An explicit value (including 0) is never overridden.
    assert _resolve_join_timing(GenerationOptions(crossfade_ms=0, sentence_pause_ms=0)) == (0, 0)
    assert _resolve_join_timing(GenerationOptions(crossfade_ms=25)) == (25, 120)
    assert _resolve_join_timing(GenerationOptions(sentence_pause_ms=90)) == (40, 90)



# ---------------------------------------------------------------------------
# Phase 12-3 output filters
# ---------------------------------------------------------------------------


def test_generation_options_output_filters_default_off():
    opts = GenerationOptions()
    assert opts.remove_dc_offset is False
    assert opts.trim_silence is False
    assert opts.fix_internal_silence is False


def test_generation_options_accepts_output_filters():
    opts = GenerationOptions(
        remove_dc_offset=True, trim_silence=True, fix_internal_silence=True
    )
    assert opts.remove_dc_offset is True
    assert opts.trim_silence is True
    assert opts.fix_internal_silence is True


def test_request_input_passes_output_filters_through():
    options = GenerationOptions(remove_dc_offset=True, trim_silence=True)
    request = TextToSpeechRequest(model="tts", input="hello", generation_options=options)
    request_input = TextToSpeechRequestInput.from_openai("tts-3", request)
    assert request_input.generation_options is options
    assert request_input.generation_options.remove_dc_offset is True
    assert request_input.generation_options.trim_silence is True
    assert request_input.generation_options.fix_internal_silence is False


def test_internal_generation_options_output_filter_defaults():
    opts = TextToSpeechGenerationOptions(seed=None, style=None, speed=None, pitch=None, energy=None)
    assert opts.remove_dc_offset is False
    assert opts.trim_silence is False
    assert opts.fix_internal_silence is False


def test_output_filters_do_not_enable_chunking():
    """The filters are post-steps; they must not change the chunking decision."""
    from harmonyspeech.common.inputs import TextToSpeechRequestInput
    from harmonyspeech.endpoints.openai.serving_text_to_speech import (
        OpenAIServingTextToSpeech,
    )

    serving = OpenAIServingTextToSpeech.__new__(OpenAIServingTextToSpeech)
    options = GenerationOptions(remove_dc_offset=True, trim_silence=True, fix_internal_silence=True)
    request = TextToSpeechRequest(model="tts", input="One short sentence.", generation_options=options)
    request_input = TextToSpeechRequestInput.from_openai("tts-4", request)
    assert serving._resolve_chunks(request_input) is None


# ---------------------------------------------------------------------------
# Phase 12-5 text-fidelity follow-up: per-model default chunk budget
# ---------------------------------------------------------------------------


def _serving_with_configs(configs):
    from types import SimpleNamespace

    from harmonyspeech.endpoints.openai.serving_text_to_speech import (
        OpenAIServingTextToSpeech,
    )

    serving = OpenAIServingTextToSpeech.__new__(OpenAIServingTextToSpeech)
    serving.engine = SimpleNamespace(engine=SimpleNamespace(model_configs=configs))
    return serving


def _budget_input(requested_model, text="One sentence here. Another one here.", **options):
    return TextToSpeechRequestInput(
        request_id="budget",
        requested_model=requested_model,
        model=requested_model,
        input_text=text,
        mode="single_speaker_tts",
        generation_options=GenerationOptions(**options),
    )


def test_default_chunk_budget_is_the_unified_300():
    from harmonyspeech.endpoints.openai.serving_text_to_speech import DEFAULT_CHUNK_SIZE

    assert DEFAULT_CHUNK_SIZE == 300
    # Phase 12-7/12-8 removed the per-model overrides; every family shares 300.
    import harmonyspeech.endpoints.openai.serving_text_to_speech as serving_module

    for removed in (
        "DEFAULT_CHUNK_SIZE_TURBO",
        "DEFAULT_CHUNK_SIZE_MULTILINGUAL",
        "DEFAULT_CHUNK_SIZE_BY_MODEL_TYPE",
    ):
        assert not hasattr(serving_module, removed), f"{removed} must be gone"


def test_turbo_pipeline_sentinel_resolves_the_300_default():
    from types import SimpleNamespace

    config = SimpleNamespace(name="chatterbox_turbo", model_type="ChatterboxTurboTTS")
    serving = _serving_with_configs([config])
    assert serving._resolve_default_chunk_size(_budget_input("chatterbox_turbo")) == 300


def test_concrete_turbo_model_name_resolves_the_300_default():
    from types import SimpleNamespace

    config = SimpleNamespace(name="my-turbo", model_type="ChatterboxTurboTTS")
    serving = _serving_with_configs([config])
    assert serving._resolve_default_chunk_size(_budget_input("my-turbo")) == 300


def test_multilingual_pipeline_sentinel_resolves_the_300_default():
    from types import SimpleNamespace

    config = SimpleNamespace(
        name="chatterbox_multilingual", model_type="ChatterboxMultilingualTTS"
    )
    serving = _serving_with_configs([config])
    assert (
        serving._resolve_default_chunk_size(_budget_input("chatterbox_multilingual"))
        == 300
    )


def test_concrete_multilingual_model_name_resolves_the_300_default():
    from types import SimpleNamespace

    config = SimpleNamespace(name="my-multilingual", model_type="ChatterboxMultilingualTTS")
    serving = _serving_with_configs([config])
    assert serving._resolve_default_chunk_size(_budget_input("my-multilingual")) == 300


def test_other_families_keep_the_global_300_default():
    from types import SimpleNamespace

    config = SimpleNamespace(name="chatterbox", model_type="ChatterboxTTS")
    serving = _serving_with_configs([config])
    assert serving._resolve_default_chunk_size(_budget_input("chatterbox")) == 300


def test_unknown_model_falls_back_to_the_global_default():
    from types import SimpleNamespace

    config = SimpleNamespace(name="chatterbox", model_type="ChatterboxTTS")
    serving = _serving_with_configs([config])
    assert serving._resolve_default_chunk_size(_budget_input("does-not-exist")) == 300


def test_bare_serving_without_an_engine_falls_back_to_the_global_default():
    from harmonyspeech.endpoints.openai.serving_text_to_speech import (
        OpenAIServingTextToSpeech,
    )

    serving = OpenAIServingTextToSpeech.__new__(OpenAIServingTextToSpeech)
    assert serving._resolve_default_chunk_size(_budget_input("chatterbox_turbo")) == 300


def test_explicit_chunk_size_always_wins_over_the_per_model_default(monkeypatch):
    """An explicit budget must bypass the per-model resolution entirely."""
    from types import SimpleNamespace

    config = SimpleNamespace(name="chatterbox_turbo", model_type="ChatterboxTurboTTS")
    serving = _serving_with_configs([config])
    text = " ".join(f"Sentence number {index} is here." for index in range(20))
    request_data = _budget_input("chatterbox_turbo", text, split_text=True, chunk_size=200)

    def _must_not_resolve(_request_data):
        raise AssertionError("explicit chunk_size must not resolve a default")

    monkeypatch.setattr(serving, "_resolve_default_chunk_size", _must_not_resolve)
    chunks = serving._resolve_chunks(request_data)
    assert chunks is not None
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_turbo_resolve_chunks_uses_the_unified_300_default():
    from types import SimpleNamespace

    config = SimpleNamespace(name="chatterbox_turbo", model_type="ChatterboxTurboTTS")
    serving = _serving_with_configs([config])
    text = " ".join(f"Sentence number {index} is here." for index in range(20))

    chunks = serving._resolve_chunks(_budget_input("chatterbox_turbo", text, split_text=True))
    assert chunks is not None and len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_multilingual_resolve_chunks_uses_the_unified_300_default():
    from types import SimpleNamespace

    config = SimpleNamespace(
        name="chatterbox_multilingual", model_type="ChatterboxMultilingualTTS"
    )
    serving = _serving_with_configs([config])
    text = " ".join(f"Sentence number {index} is here." for index in range(20))

    chunks = serving._resolve_chunks(
        _budget_input("chatterbox_multilingual", text, split_text=True)
    )
    assert chunks is not None and len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)

