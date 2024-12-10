import { DefaultApi, ApiClient } from "@harmony-ai/harmonyspeech";
import {
    VoiceConversionRequest,
    VoiceConversionResponse,
    EmbedSpeakerRequest,
    EmbedSpeakerResponse,
    TextToSpeechRequest,
    TextToSpeechResponse,
    SpeechTranscribeRequest,
    SpeechToTextResponse,
    ModelList
} from "@harmony-ai/harmonyspeech";

class HarmonySpeechEnginePlugin {
    constructor(apiKey = '', baseURL = 'http://localhost:12080') {
        this.apiKey = apiKey;
        this.apiClient = new ApiClient(baseURL);
        this.defaultApi = new DefaultApi(this.apiClient);
    }

    /**
     * Update the API endpoint (baseURL) after initialization.
     * @param {String} newBaseURL - The new base URL to set.
     */
    setBaseURL(newBaseURL) {
        if (!newBaseURL || typeof newBaseURL !== 'string') {
            throw new Error('Invalid newBaseURL parameter');
        }
        this.apiClient.basePath = newBaseURL;
    }

    /**
     * Update the API key after initialization.
     * @param {String} newApiKey - The new API key to set.
     */
    setApiKey(newApiKey) {
        if (!newApiKey || typeof newApiKey !== 'string') {
            throw new Error('Invalid newApiKey parameter');
        }
        this.apiKey = newApiKey;
    }

    /**
     * Error handler to parse and throw errors with meaningful messages.
     * @param {String} message - Custom error message.
     * @param {Error} error - Original error object.
     * @throws {Error} - Throws a new error with detailed information.
     */
    handleError(message, error) {
        if (error.response && error.response.body) {
            console.error(message, error.response.body);
            throw new Error(`${message} ${JSON.stringify(error.response.body)}`);
        } else if (error.message) {
            console.error(message, error.message);
            throw new Error(`${message} ${error.message}`);
        } else {
            console.error(message, error);
            throw new Error(`${message} ${error}`);
        }
    }

    /**
     * Convert voice
     * @param {VoiceConversionRequest} voiceConversionRequest - The request payload.
     * @param {Object} [options] - Optional parameters.
     * @returns {Promise<VoiceConversionResponse>} - The response object.
     * @throws {Error} - Throws an error if the input is invalid or the API call fails.
     */
    async convertVoice(voiceConversionRequest, options = {}) {
        if (!voiceConversionRequest || typeof voiceConversionRequest !== 'object') {
            throw new Error('Invalid voiceConversionRequest parameter');
        }
        if (this.apiKey && (options['xApiKey'] === undefined || options['xApiKey'] === null)) {
            options['xApiKey'] = this.apiKey;
        }
        try {
            return await this.defaultApi.convertVoiceV1VoiceConvertPost(voiceConversionRequest, options);
        } catch (error) {
            this.handleError('Error converting voice:', error);
        }
    }

    /**
     * Create Embedding
     * @param {EmbedSpeakerRequest} embedSpeakerRequest - The request payload.
     * @param {Object} [options] - Optional parameters.
     * @returns {Promise<EmbedSpeakerResponse>} - The response object.
     * @throws {Error} - Throws an error if the input is invalid or the API call fails.
     */
    async createEmbedding(embedSpeakerRequest, options = {}) {
        if (!embedSpeakerRequest || typeof embedSpeakerRequest !== 'object') {
            throw new Error('Invalid embedSpeakerRequest parameter');
        }
        if (this.apiKey && (options['xApiKey'] === undefined || options['xApiKey'] === null)) {
            options['xApiKey'] = this.apiKey;
        }
        try {
            return await this.defaultApi.createEmbeddingV1EmbedSpeakerPost(embedSpeakerRequest, options);
        } catch (error) {
            this.handleError('Error creating embedding:', error);
        }
    }

    /**
     * Create speech
     * @param {TextToSpeechRequest} textToSpeechRequest - The request payload.
     * @param {Object} [options] - Optional parameters, such as headers.
     * @returns {Promise<TextToSpeechResponse>} - The response object.
     * @throws {Error} - Throws an error if the input is invalid or the API call fails.
     */
    async createSpeech(textToSpeechRequest, options = {}) {
        if (!textToSpeechRequest || typeof textToSpeechRequest !== 'object') {
            throw new Error('Invalid textToSpeechRequest parameter');
        }
        if (this.apiKey && (options['xApiKey'] === undefined || options['xApiKey'] === null)) {
            options['xApiKey'] = this.apiKey;
        }
        try {
            return await this.defaultApi.createSpeechV1AudioSpeechPost(textToSpeechRequest, options);
        } catch (error) {
            this.handleError('Error creating speech:', error);
        }
    }

    /**
     * Create transcription
     * @param {SpeechTranscribeRequest} speechTranscribeRequest - The request payload.
     * @param {Object} [options] - Optional parameters, such as headers.
     * @returns {Promise<SpeechToTextResponse>} - The response object.
     * @throws {Error} - Throws an error if the input is invalid or the API call fails.
     */
    async createTranscription(speechTranscribeRequest, options = {}) {
        if (!speechTranscribeRequest || typeof speechTranscribeRequest !== 'object') {
            throw new Error('Invalid speechTranscribeRequest parameter');
        }
        if (this.apiKey && (options['xApiKey'] === undefined || options['xApiKey'] === null)) {
            options['xApiKey'] = this.apiKey;
        }
        try {
            return await this.defaultApi.createTranscriptionV1AudioTranscriptionsPost(speechTranscribeRequest, options);
        } catch (error) {
            this.handleError('Error creating transcription:', error);
        }
    }

    /**
     * Check health of the API service.
     * @param {Object} [options] - Optional parameters, such as headers.
     * @returns {Promise<Object>} - The response object indicating the health status.
     * @throws {Error} - Throws an error if the API call fails.
     */
    async checkHealth(options = {}) {
        if (this.apiKey && (options['xApiKey'] === undefined || options['xApiKey'] === null)) {
            options['xApiKey'] = this.apiKey;
        }
        try {
            return await this.defaultApi.healthHealthGet(options);
        } catch (error) {
            this.handleError('Error checking health:', error);
        }
    }

    /**
     * Fetch available models for embedding creation.
     * @param {Object} [options] - Optional parameters, such as headers.
     * @returns {Promise<ModelList>} - The list of available embedding models.
     * @throws {Error} - Throws an error if the API call fails.
     */
    async showAvailableEmbeddingModels(options = {}) {
        if (this.apiKey && (options['xApiKey'] === undefined || options['xApiKey'] === null)) {
            options['xApiKey'] = this.apiKey;
        }
        try {
            return await this.defaultApi.showAvailableEmbeddingModelsV1EmbedModelsGet(options);
        } catch (error) {
            this.handleError('Error fetching available embedding models:', error);
        }
    }

    /**
     * Fetch available speech synthesis models.
     * @param {Object} [options] - Optional parameters, such as headers.
     * @returns {Promise<ModelList>} - The list of available speech models.
     * @throws {Error} - Throws an error if the API call fails.
     */
    async showAvailableSpeechModels(options = {}) {
        if (this.apiKey && (options['xApiKey'] === undefined || options['xApiKey'] === null)) {
            options['xApiKey'] = this.apiKey;
        }
        try {
            return await this.defaultApi.showAvailableSpeechModelsV1AudioSpeechModelsGet(options);
        } catch (error) {
            this.handleError('Error fetching available speech models:', error);
        }
    }

    /**
     * Fetch available models for transcription.
     * @param {Object} [options] - Optional parameters, such as headers.
     * @returns {Promise<ModelList>} - The list of available transcription models.
     * @throws {Error} - Throws an error if the API call fails.
     */
    async showAvailableTranscriptionModels(options = {}) {
        if (this.apiKey && (options['xApiKey'] === undefined || options['xApiKey'] === null)) {
            options['xApiKey'] = this.apiKey;
        }
        try {
            return await this.defaultApi.showAvailableTranscriptionModelsV1AudioTranscriptionsModelsGet(options);
        } catch (error) {
            this.handleError('Error fetching available transcription models:', error);
        }
    }

    /**
     * Fetch available models for voice conversion.
     * @param {Object} [options] - Optional parameters, such as headers.
     * @returns {Promise<ModelList>} - The list of available voice conversion models.
     * @throws {Error} - Throws an error if the API call fails.
     */
    async showAvailableVoiceConversionModels(options = {}) {
        if (this.apiKey && (options['xApiKey'] === undefined || options['xApiKey'] === null)) {
            options['xApiKey'] = this.apiKey;
        }
        try {
            return await this.defaultApi.showAvailableVoiceConversionModelsV1VoiceConvertModelsGet(options);
        } catch (error) {
            this.handleError('Error fetching available voice conversion models:', error);
        }
    }

    /**
     * Fetch the version of the Harmony Speech Engine API.
     * @param {Object} [options] - Optional parameters, such as headers.
     * @returns {Promise<Object>} - The API version information.
     * @throws {Error} - Throws an error if the API call fails.
     */
    async showVersion(options = {}) {
        if (this.apiKey && (options['xApiKey'] === undefined || options['xApiKey'] === null)) {
            options['xApiKey'] = this.apiKey;
        }
        try {
            return await this.defaultApi.showVersionVersionGet(options);
        } catch (error) {
            this.handleError('Error fetching version:', error);
        }
    }
}

export default HarmonySpeechEnginePlugin;