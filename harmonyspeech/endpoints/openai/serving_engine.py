import asyncio
import copy
import json
from http import HTTPStatus

from harmonyspeech.common.config import ModelConfig
from harmonyspeech.endpoints.openai.protocol import (
    ErrorResponse,
    LanguageOptions,
    ModelCard,
    ModelList,
    TextToSpeechRequest,
    VoiceOptions,
)
from harmonyspeech.engine.async_harmonyspeech import AsyncHarmonySpeech


class OpenAIServing:
    def __init__(self, engine: AsyncHarmonySpeech, available_models: list[ModelCard]):
        self.engine = engine
        self.available_models = available_models

        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        if event_loop is not None and event_loop.is_running():
            # If the current is instanced by Ray Serve,
            # there is already a running event loop
            event_loop.create_task(self._post_init())
        else:
            # When using single HarmonySpeech without engine_use_ray
            asyncio.run(self._post_init())

    async def _post_init(self):
        engine_model_config = await self.engine.get_model_configs()

    async def show_available_models(self) -> ModelList:
        return ModelList(data=self.available_models)

    def create_error_response(
        self, message: str, err_type: str = "BadRequestError", status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
    ) -> ErrorResponse:
        return ErrorResponse(message=message, type=err_type, code=status_code.value)

    def create_streaming_error_response(
        self, message: str, err_type: str = "BadRequestError", status_code: HTTPStatus = HTTPStatus.BAD_REQUEST
    ) -> str:
        json_str = json.dumps(
            {
                "error": self.create_error_response(
                    message=message, err_type=err_type, status_code=status_code
                ).model_dump()
            }
        )
        return json_str

    def get_model_by_id(self, model: str) -> ModelCard | None:
        for model_card in self.available_models:
            if model_card.id == model:
                return model_card
        return None

    # Chatterbox model IDs that support both single_speaker_tts and voice_cloning
    _CHATTERBOX_MODEL_IDS = {"chatterbox", "chatterbox_turbo", "chatterbox_multilingual"}

    async def _check_model(self, request) -> ErrorResponse | None:
        if isinstance(request, TextToSpeechRequest) and request.mode not in ["voice_cloning", "single_speaker_tts"]:
            # TODO: Evaluate this based on model toolchain capability
            return self.create_error_response(
                message="Parameter `mode` must either be 'single_speaker_tts' or 'voice_cloning'.",
                err_type="BadRequestError",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        # Validate mode vs input_audio/input_embedding consistency for Chatterbox models
        if isinstance(request, TextToSpeechRequest) and request.model in self._CHATTERBOX_MODEL_IDS:
            has_reference = request.input_audio is not None or request.input_embedding is not None
            if request.mode == "voice_cloning" and not has_reference:
                return self.create_error_response(
                    message=f"Chatterbox model '{request.model}' with mode 'voice_cloning' requires either `input_audio` (reference audio) or `input_embedding` (pre-computed embedding).",
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            if request.mode == "single_speaker_tts" and has_reference:
                return self.create_error_response(
                    message=f"Chatterbox model '{request.model}' with mode 'single_speaker_tts' should not have `input_audio` or `input_embedding`. Use 'voice_cloning' mode for voice cloning.",
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )

        model = self.get_model_by_id(request.model)
        if model is None:
            return self.create_error_response(
                message=f"The model `{request.model}` does not exist.",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND,
            )

        # Checks for Language and Voice parameters if the models have these.
        # Only validate when the request type actually has a 'language' field
        # (EmbedSpeakerRequest and similar endpoint requests do not).
        if model.languages and len(model.languages) > 0 and hasattr(request, "language"):
            if not request.language:
                return self.create_error_response(
                    message=f"The model `{request.model}` requires a language parameter.",
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            allowed_languages = [l.language for l in model.languages]
            if request.language not in allowed_languages:
                return self.create_error_response(
                    message=f"The model `{request.model}` only supports the following languages: {','.join(allowed_languages)}.",
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )

            # Voice ID is always a subset of the language specified
            language_option = next((lang for lang in model.languages if lang.language == request.language), None)
            if language_option is None:
                return self.create_error_response(
                    message="Issue while retrieving language option for the model.",
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            if language_option.voices and len(language_option.voices) > 0:
                if not request.voice:
                    return self.create_error_response(
                        message=f"The model `{request.model}` requires a voice parameter for language `{request.language}`.",
                        err_type="BadRequestError",
                        status_code=HTTPStatus.BAD_REQUEST,
                    )

                allowed_voices = [v.voice for v in language_option.voices]
                if request.voice not in allowed_voices:
                    return self.create_error_response(
                        message=f"The model `{request.model}` only supports the following voices for language `{request.language}`: {','.join(allowed_voices)}.",
                        err_type="BadRequestError",
                        status_code=HTTPStatus.BAD_REQUEST,
                    )
                voice_option = next((voice for voice in language_option.voices if voice.voice == request.voice), None)
                if voice_option is None:
                    return self.create_error_response(
                        message="Issue while processing the voice option for the model.",
                        err_type="InternalServerError",
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )

    @staticmethod
    def model_card_from_config(config: ModelConfig) -> ModelCard:
        card = ModelCard(id=config.name, root=config.name)
        if config.language is not None and config.language != "":
            lang_option = LanguageOptions(language=config.language)
            if card.languages is None:
                card.languages = []
            card.languages.append(lang_option)
            if config.voices is not None and len(config.voices) > 0:
                for voice in config.voices:
                    voice_option = VoiceOptions(voice=voice)
                    if lang_option.voices is None:
                        lang_option.voices = []
                    lang_option.voices.append(voice_option)
        elif config.voices is not None and len(config.voices) > 0:
            # Handle models with voices but no specific language (e.g. KittenTTS)
            # We map these under a "default" language so they show up in the API and UI
            lang_option = LanguageOptions(language="default")
            card.languages = [lang_option]
            for voice in config.voices:
                voice_option = VoiceOptions(voice=voice)
                if lang_option.voices is None:
                    lang_option.voices = []
                lang_option.voices.append(voice_option)

        # Add ChatterboxMultilingualTTS language support (23 languages)
        if config.model_type == "ChatterboxMultilingualTTS":
            from harmonyspeech.modeling.models.chatterbox.chatterbox import ChatterboxMultilingualTTSModel

            for lang_code, lang_name in ChatterboxMultilingualTTSModel.SUPPORTED_LANGUAGES.items():
                lang_option = LanguageOptions(language=lang_code)
                if card.languages is None:
                    card.languages = []
                card.languages.append(lang_option)

        return card

    @staticmethod
    def model_card_add_config(card: ModelCard, config: ModelConfig):
        # Check for language - Merge language options if possible
        if config.language is not None and config.language != "":
            lang_option = None
            if card.languages is not None:
                for option in card.languages:
                    if option.language == config.language:
                        lang_option = option
                        break

            if lang_option is None:
                lang_option = LanguageOptions(language=config.language)
                if card.languages is None:
                    card.languages = []
                card.languages.append(lang_option)

            # Voices per language - merge voice options if possible
            if config.voices is not None and len(config.voices) > 0:
                for voice in config.voices:
                    voice_option = None
                    if lang_option.voices is not None:
                        for option in lang_option.voices:
                            if option.voice == voice:
                                voice_option = option
                                break
                    if voice_option is None:
                        voice_option = VoiceOptions(voice=voice)
                        if lang_option.voices is None:
                            lang_option.voices = []
                        lang_option.voices.append(voice_option)

    @staticmethod
    def model_cards_from_config_groups(configured_models, individual_model_types, model_groups):
        model_cards = []
        individual_model_ids = set()
        # create a copy of the model groups and remove each model_type of a group from it's list if an instance exits
        # only groups where all required model types to provide the API function exist will be enabled.
        check_dict = copy.deepcopy(model_groups)
        for m in configured_models:
            if m.model_type in individual_model_types:
                # Retrieve model card from card list if existing
                model_card = None
                for c in model_cards:
                    if c.id == m.name:
                        model_card = c
                        break

                if model_card is None:
                    model_card = OpenAIServing.model_card_from_config(m)
                    model_cards.append(model_card)

                individual_model_ids.add(m.name)

            for group_name, remaining_models in check_dict.items():
                # Create or extend card for group
                if m.model_type in model_groups[group_name]:
                    # Retrieve model card from card list if existing
                    group_card = None
                    for c in model_cards:
                        if c.id == group_name:
                            group_card = c
                            break
                    if group_card is None:
                        group_card = OpenAIServing.model_card_from_config(m)
                        group_card.id = group_name
                        group_card.root = group_name
                        group_card.object = "toolchain"
                        model_cards.append(group_card)
                    else:
                        OpenAIServing.model_card_add_config(group_card, m)

                # Check if group init models remaining and whether a model of this type is required for init
                if len(remaining_models) == 0:
                    continue
                if m.model_type in remaining_models:
                    # Remove model from check dict
                    remaining_models.remove(m.model_type)

        # Safety check!
        # Only allow model groups which have the required models initialized
        # BUT: do not filter out models that were also registered as individual models
        groups_with_remaining = [group_name for group_name, remaining_models in check_dict.items() if remaining_models]
        model_cards = [
            card for card in model_cards if card.id not in groups_with_remaining or card.id in individual_model_ids
        ]
        return model_cards
