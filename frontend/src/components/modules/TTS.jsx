import {useEffect, useState} from "react";
import SettingsTooltip from "../settings/SettingsTooltip.jsx";
import HarmonySpeechEnginePlugin from "../../plugins/HarmonySpeech.js";
import HarmonyAudioPlayer from "../widgets/HarmonyAudioPlayer.jsx";
import Heatmap from "../widgets/Heatmap.jsx";

const knownModelNames = {
    "harmonyspeech": "HarmonySpeech V1",
    "openvoice_v1": "OpenVoice V1",
    "openvoice_v2": "OpenVoice V2",
}

// Embedding Status values
const embeddingStatusNone = "No embedding loaded.";
const embeddingStatusInProgress = "Embedding in progress...";
const embeddingStatusDone = "Embedding updated.";
const embeddingStatusFailed = "Embedding failed.";

// Voice Config using local storage; mimicing Wails FS functionality used in Harmony Link
const ListVoiceConfigs = () => {
  // Get keys from localStorage that start with 'voiceConfig_'
  const configs = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key.startsWith("voiceConfig_")) {
      configs.push(key.substring("voiceConfig_".length));
    }
  }
  return Promise.resolve(configs);
};

const LoadVoiceConfigFromFile = (configName) => {
  const configString = localStorage.getItem("voiceConfig_" + configName);
  return Promise.resolve(configString);
};

const SaveVoiceConfigToFile = (configName, configString) => {
  localStorage.setItem("voiceConfig_" + configName, configString);
  return Promise.resolve();
};


const TTS = ({initialSettings}) => {
    const [tooltipVisible, setTooltipVisible] = useState(0);

    // Modal dialog values
    const [modalTitle, setModalTitle] = useState('Invalid Input');
    const [modalMessage, setModalMessage] = useState('');
    const [isModalVisible, setIsModalVisible] = useState(false);

    const [confirmModalVisible, setConfirmModalVisible] = useState(false);
    const [confirmModalTitle, setConfirmModalTitle] = useState('Confirmation required');
    const [confirmModalMessage, setConfirmModalMessage] = useState('');
    const [confirmModalYes, setConfirmModalYes] = useState((confirmValue)=>{});
    const [confirmModalNo, setConfirmModalNo] = useState(()=>{});
    const [confirmModalHasInput, setConfirmModalHasInput] = useState(false);
    const [confirmModalInput, setConfirmModalInput] = useState('');

    // Show Modal Functions
    const showModal = (message, title='Invalid Input') => {
        setModalTitle(title);
        setModalMessage(message);
        setIsModalVisible(true);
    };

    const showConfirmModal = (message, hasInput=false, title='Confirmation required') => {
        setConfirmModalTitle(title);
        setConfirmModalMessage(message);
        setConfirmModalHasInput(hasInput);
        setConfirmModalVisible(true);
    };

    // Base Settings reference
    const [moduleSettings, setModuleSettings] = useState(initialSettings);

    // Harmonyspeech Plugin
    const [harmonySpeechPlugin, setHarmonySpeechPlugin] = useState(null);

    // model, language and voice options dynamically fetched from HSE
    const [modelOptions, setModelOptions] = useState([
        {name: "Error: no models available", value: null}
    ]);
    const [modelLanguageOptions, setModelLanguageOptions] = useState({});
    const [modelVoiceOptions, setModelVoiceOptions] = useState({});

    // TODO: Fetch operation modes dynamically from HSE
    const modelOperationModes = {
        'harmonyspeech': [
            {name: 'Voice Cloning', value: 'voice_cloning'},
        ],
        'openvoice_v1': [
            {name: 'Single-Speaker TTS', value: 'single_speaker_tts'},
            {name: 'Voice Cloning', value: 'voice_cloning'},
        ],
        'openvoice_v2': [
            {name: 'Single-Speaker TTS', value: 'single_speaker_tts'},
            {name: 'Voice Cloning', value: 'voice_cloning'},
        ]
    }

    // Internal State handling
    const [voiceConfigs, setVoiceConfigs] = useState([]);
    const [currentVoiceConfig, setCurrentVoiceConfig] = useState({
        // Basic Settings
        model: "harmonyspeech",
        operation_mode: "voice_cloning",
        language: "",
        voice: "",
        style: 0,
        speed: 1.00,
        pitch: 1.00,
        energy: 1.00,
        seed: 42,
        // Embedding Data
        // source_embedding: "",
        target_embedding: ""
    });

    // Setting Fields
    const [endpoint, setEndpoint] = useState(initialSettings.endpoint);
    const [voiceConfigFile, setVoiceConfigFile] = useState(initialSettings.voiceconfigfile);

    // Voice Embedding States
    const [embeddingFile, setEmbeddingFile] = useState(null);
    const [embeddingFileAudio, setEmbeddingFileAudio] = useState(null);
    const [embeddingStatus, setEmbeddingStatus] = useState(embeddingStatusNone);

    // Testing Area States
    const [generationText, setGenerationText] = useState("This is a sample text");
    const [generatedAudio, setGeneratedAudio] = useState(null);

    // Validation Functions
    const validateEndpointAndUpdate = (value) => {
        const urlRegex = /^(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?([a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}|localhost|\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3})(:[0-9]{1,5})?(\/.*)?$/;
        if ((moduleSettings.endpoint.length > 0 && value.length === 0) || (value.length > 0 && urlRegex.test(value) === false)) {
            showModal("Endpoint URL must be a valid URL.");
            setEndpoint(moduleSettings.endpoint);
            return false;
        }
        // Update if validation successful
        moduleSettings.endpoint = value;

        // Refresh Speech Tooling
        harmonySpeechPlugin.setBaseURL(value);
        refreshAvailableTTSToolchains(harmonySpeechPlugin);
        return true;
    };

    // Utility Functions
    const setupHarmonySpeechTooling = () => {
        const plugin = new HarmonySpeechEnginePlugin(moduleSettings.userapikey, moduleSettings.endpoint);
        setHarmonySpeechPlugin(plugin);

        // Fetch available toolchains from Endpoint (if available)
        refreshAvailableTTSToolchains(plugin);
    }

    const refreshVoiceConfigs = () => {
        try {
            ListVoiceConfigs().then((result) => {
                setVoiceConfigs(result);
                if (result.length > 0) {
                    changeVoiceConfigAndUpdate(result[0]);
                }
            });
        } catch (error) {
            console.error("Unable to load available voice configurations");
            console.error(error);
            showModal("Error loading voice config", "An Error occurred");
        }
    }

    const refreshAvailableTTSToolchains = (harmonySpeechClient) => {
        if (!harmonySpeechClient) {
            console.error("Harmony Speech Client not initialized");
            return;
        }
        harmonySpeechClient.showAvailableSpeechModels().then((result) => {
            //console.log(JSON.stringify(result.data));

            // Search for Toolchains and add them to the list
            const newModelOptions = [];
            const newModelLanguageOptions = {};
            const newModelVoiceOptions = {};
            result.data.forEach((model) => {
                if (model.object === "toolchain") {
                    if (model.id in knownModelNames) {
                        newModelOptions.push({name: knownModelNames[model.id], value: model.id});
                    } else {
                        newModelOptions.push({name: model.id, value: model.id});
                    }

                    // Fetch language options
                    // If values are not set for a combination, UI needs to show default value (see UI code further below)
                    if (model.languages.length > 0) {
                        if (!newModelLanguageOptions[model.id]) {
                            newModelLanguageOptions[model.id] = []
                        }
                        model.languages.forEach((langOption) => {
                            newModelLanguageOptions[model.id].push({name: langOption.language, value: langOption.language})
                            // Fetch voice option for language
                            if (langOption.voices.length > 0) {
                                if (!newModelVoiceOptions[model.id]) {
                                    newModelVoiceOptions[model.id] = {}
                                }
                                langOption.voices.forEach((voiceOption) => {
                                    if (!newModelVoiceOptions[model.id][langOption.language]) {
                                        newModelVoiceOptions[model.id][langOption.language] = []
                                    }
                                    newModelVoiceOptions[model.id][langOption.language].push({name: voiceOption.voice, value: voiceOption.voice})
                                });
                            }
                        });
                    }
                }
            });
            if (newModelOptions.length === 0) {
                newModelOptions.push({name: "Error: no models available", value: null});
            }
            setModelOptions(newModelOptions);
            setModelLanguageOptions(newModelLanguageOptions);
            setModelVoiceOptions(newModelVoiceOptions);

            // Refresh UI
            if(!newModelOptions.some((modelOption) => modelOption.value === currentVoiceConfig.model)) {
                handleModelSelectionChange(newModelOptions[0].value);
            }
        });
    }

    const setInitialValues = () => {
        // Reset Entity map
        setModuleSettings(initialSettings);

        // Setup Harmony Speech
        setupHarmonySpeechTooling();

        // Update Voice Configs
        refreshVoiceConfigs();
    };

    useEffect(() => {
        console.log(JSON.stringify(initialSettings));
        setInitialValues();
    }, []);

    // Config Management Handlers
    const changeVoiceConfigAndUpdate = async (selectedConfig) => {
        if (selectedConfig === "") {
            return false;
        }

        try {
            // Parse Data and Update current config data
            const configData = await LoadVoiceConfigFromFile(selectedConfig);
            const parsedConfig = JSON.parse(configData);
            setCurrentVoiceConfig(parsedConfig);
            // Update Harmony Link Settings
            setVoiceConfigFile(selectedConfig);
            moduleSettings.voiceconfigfile = selectedConfig;
            return true;
        } catch (error) {
            console.error("Failed to load voice configuration.");
            console.error(error);
            showModal("Failed to load the selected voice configuration.", "An Error occurred");
            return false;
        }
    };

    const saveVoiceConfiguration = (configName) => {
        if (!configName) {
            showModal("No config name provided");
            return false;
        }

        try {
            const configString = JSON.stringify(currentVoiceConfig, null, 2);
            SaveVoiceConfigToFile(configName, configString)
                .then(() => {
                    refreshVoiceConfigs();
                    showModal("Configuration saved successfully.", "Success");
                    setConfirmModalVisible(false);
                })
                .catch((error) => {
                    console.error("Failed to save voice configuration.");
                    console.error(error);
                    showModal("Failed to save the voice configuration.", "An Error occurred");
                });
            return true;
        } catch (error) {
            console.error("Error stringifying the voice configuration.");
            console.error(error);
            showModal("Failed to save the voice configuration.", "An Error occurred");
            return false;
        }
    }

    const cancelSaveVoiceConfiguration = () => {
        setConfirmModalInput('');
        setConfirmModalVisible(false);
    }

    const handleSaveConfig = () => {
        setConfirmModalInput('');
        setConfirmModalYes(() => saveVoiceConfiguration);
        setConfirmModalNo(() => cancelSaveVoiceConfiguration);
        showConfirmModal("Please enter a name for the configuration", true, "Save voice configuration");
    };

    const handleVoiceSelectionChange = (selectedVoiceId) => {
        // Ensure selected voice is valid
        let voice = selectedVoiceId;
        if (!modelVoiceOptions[currentVoiceConfig.model] || !modelVoiceOptions[currentVoiceConfig.model][currentVoiceConfig.language] || modelVoiceOptions[currentVoiceConfig.model][currentVoiceConfig.language].length === 0) {
            // Model has no defined voices
            voice = "";
        } else if (!modelVoiceOptions[currentVoiceConfig.model][currentVoiceConfig.language].some((voiceOption) => voiceOption.value === voice)) {
            // Model has voices, but selected one is not a valid one
            voice = modelVoiceOptions[currentVoiceConfig.model][currentVoiceConfig.language][0].value;
        }

        // Update config
        const newConfig = {
            ...currentVoiceConfig,
            voice: voice,
        };
        setCurrentVoiceConfig(newConfig);
    };

    const handleLanguageSelectionChange = (selectedLanguageId) => {
        // Ensure selected language is valid
        let language = selectedLanguageId;
        if (!modelLanguageOptions[currentVoiceConfig.model] || modelLanguageOptions[currentVoiceConfig.model].length === 0) {
            // Model has no valid languages
            language = ""
        } else if (!modelLanguageOptions[currentVoiceConfig.model].some((langOption) => langOption.value === language)) {
            // Model has languages, but selected one is not a valid one
            language = modelLanguageOptions[currentVoiceConfig.model][0].value;
        }

        // Ensure selected voice is valid
        let voice = currentVoiceConfig.voice;
        if (!modelVoiceOptions[currentVoiceConfig.model] || !modelVoiceOptions[currentVoiceConfig.model][language] || modelVoiceOptions[currentVoiceConfig.model][language].length === 0) {
            // Model has no defined voices
            voice = "";
        } else if (!modelVoiceOptions[currentVoiceConfig.model][language].some((voiceOption) => voiceOption.value === voice)) {
            // Model has voices, but selected one is not a valid one
            voice = modelVoiceOptions[currentVoiceConfig.model][language][0].value;
        }

        // Update config
        const newConfig = {
            ...currentVoiceConfig,
            language: language,
            voice: voice,
        };
        setCurrentVoiceConfig(newConfig);
    };

    const handleModelSelectionChange = (selectedModelId) => {
        // Ensure selected OperationMode is valid
        let operationMode = currentVoiceConfig.operation_mode;
        if (!modelOperationModes[selectedModelId].some((mode) => mode.value === currentVoiceConfig.operation_mode)) {
            operationMode = modelOperationModes[selectedModelId][0].value;
        }

        // Ensure selected language is valid
        let language = currentVoiceConfig.language;
        if (!modelLanguageOptions[selectedModelId] || modelLanguageOptions[selectedModelId].length === 0) {
            // Model has no valid languages
            language = ""
        } else if (!modelLanguageOptions[selectedModelId].some((langOption) => langOption.value === language)) {
            // Model has languages, but selected one is not a valid one
            language = modelLanguageOptions[selectedModelId][0].value;
        }

        // Ensure selected voice is valid
        let voice = currentVoiceConfig.voice;
        if (!modelVoiceOptions[selectedModelId] || !modelVoiceOptions[selectedModelId][language] || modelVoiceOptions[selectedModelId][language].length === 0) {
            // Model has no defined voices
            voice = "";
        } else if (!modelVoiceOptions[selectedModelId][language].some((voiceOption) => voiceOption.value === voice)) {
            // Model has voices, but selected one is not a valid one
            voice = modelVoiceOptions[selectedModelId][language][0].value;
        }

        // Update config
        const newConfig = {
            ...currentVoiceConfig,
            model: selectedModelId,
            operation_mode: operationMode,
            language: language,
            voice: voice,
            target_embedding: "",
        };
        setCurrentVoiceConfig(newConfig);
        setEmbeddingStatus(embeddingStatusNone);
        //console.log(JSON.stringify(newConfig));
    }

    const handleOperationModeChange = (selectedOperationMode) => {
        setCurrentVoiceConfig({
            ...currentVoiceConfig,
            operation_mode: selectedOperationMode,
            target_embedding: "",
        });
        setEmbeddingStatus(embeddingStatusNone);
    }

    const handleEmbeddingFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setEmbeddingFile(file);
            // Load into Player
            const reader = new FileReader();
            reader.onload = async () => {
                const dataUrl = reader.result;
                setEmbeddingFileAudio(dataUrl);
            }
            reader.readAsDataURL(file);
        }
    };

    const handleGenerateEmbedding = async () => {
        if (!embeddingFile) {
            showModal("Please select a voice file to embed.");
            return;
        }

        const reader = new FileReader();
        reader.onload = async () => {
            const dataUrl = reader.result;
            const base64Content = dataUrl.split(',')[1];
            try {
                setEmbeddingStatus(embeddingStatusInProgress);
                const response = await harmonySpeechPlugin.createEmbedding({
                    model: currentVoiceConfig.model,
                    input_audio: base64Content,
                });
                setCurrentVoiceConfig((prev) => ({
                    ...prev,
                    target_embedding: response.data,
                }));
                setEmbeddingStatus(embeddingStatusDone);
                console.log(JSON.stringify(currentVoiceConfig));
            } catch (error) {
                console.error("Embedding request failed.");
                console.error(error);
                setEmbeddingStatus(embeddingStatusFailed);
                showModal("Failed to perform embedding: " + error.message, "An Error occurred");
            }
        };
        reader.onerror = () => {
            showModal("Failed to read the selected file.", "An Error occurred");
        };
        reader.readAsDataURL(embeddingFile);
    };

    // Voice Synthesis Handlers
    const handleSynthesizeVoice = async () => {
        if (!generationText) {
            showModal("Please enter text to synthesize.");
            return;
        }
        if (currentVoiceConfig.operation_mode === "voice_cloning" && !currentVoiceConfig.target_embedding) {
            showModal("Please generate an embedding for voice cloning first");
            return;
        }

        try {
            // This sends a TTS Request
            // For voice cloning, we're assuming there has been an embedding generated already.
            // Cloning a voice starting from an existing embedding is always more performant than doing dynamic cloning.
            const response = await harmonySpeechPlugin.createSpeech({
                // Basic config
                model: currentVoiceConfig.model,
                input: generationText,
                mode: currentVoiceConfig.operation_mode,
                language: currentVoiceConfig.language,
                voice: currentVoiceConfig.voice,
                input_embedding: currentVoiceConfig.target_embedding ? currentVoiceConfig.target_embedding : null,
                // Generation Options
                generation_options: {
                    seed: currentVoiceConfig.seed,
                    style: currentVoiceConfig.style,
                    speed: currentVoiceConfig.speed,
                    pitch: currentVoiceConfig.pitch,
                    energy: currentVoiceConfig.energy,
                },
                // TODO: Output Options
                // TODO: Post generation filters
            });
            setGeneratedAudio(`data:audio/wav;base64,${response.data}`);
        } catch (error) {
            console.error("Text-to-Speech request failed.");
            console.error(error);
            showModal("Failed to generate speech: " + error.message, "An Error occurred");
        }
    };

    return (
        <>
            <div className="flex flex-wrap w-full pt-2">
                {/*<label className="bold italic text-orange-400">*/}
                {/*    A better config UI which allows for creating custom voices Harmony Speech V1 locally coming soon.*/}
                {/*    <br/>Harmony Speech V1 models will be released Open Source for local usage soon!*/}
                {/*    <br/>For more infos, please join our <span className="text-blue-500 font-bold"><a*/}
                {/*    href="https://discord.gg/f6RQyhNPX8" target="_blank">Discord server</a></span>.*/}
                {/*</label>*/}
                <div className="flex flex-wrap items-center -px-10 w-full">
                    <div className="flex items-center mb-6 w-full">
                        <div className="flex items-center mt-2 w-full">
                            <label className="block text-sm font-medium text-gray-300 w-1/3 px-3">
                                Voice Configuration
                                <SettingsTooltip tooltipIndex={1} tooltipVisible={() => tooltipVisible}
                                                 setTooltipVisible={setTooltipVisible}>
                                    Select from existing voice configurations or save the current config
                                </SettingsTooltip>
                            </label>
                            <select
                                value={voiceConfigFile}
                                onChange={(e) => changeVoiceConfigAndUpdate(e.target.value)}
                                className="block w-1/3 bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                            >
                                {voiceConfigs && voiceConfigs.length > 0 ? (
                                    voiceConfigs.map((config) => (
                                        <option key={config} value={config}>
                                            {config}
                                        </option>
                                    ))
                                ) : (
                                    <option value="">No Configs Available</option>
                                )}
                            </select>
                            <button onClick={handleSaveConfig}
                                    className="bg-neutral-700 hover:bg-neutral-500 font-semibold py-1 px-2 mx-1 text-orange-400 text-sm">
                                Save current settings
                            </button>
                        </div>
                    </div>

                    <div className="flex flex-wrap items-center mb-6 w-full border-t border-neutral-500">
                        <div className="flex items-center mt-2 mb-2 w-full">
                            <h2 className="text-l font-bold text-gray-300">Basic Settings</h2>
                        </div>
                        <div className="flex items-center mb-6 w-full">
                            <div className="flex items-center w-full">
                                <label className="block text-sm font-medium text-gray-300 w-1/4 px-3">
                                    Endpoint URL
                                    <SettingsTooltip tooltipIndex={2} tooltipVisible={() => tooltipVisible}
                                                     setTooltipVisible={setTooltipVisible}>
                                        Specify the endpoint for the Harmony Speech Engine API.
                                    </SettingsTooltip>
                                </label>
                                <div className="w-3/4">
                                    <input type="text" name="endpoint"
                                           className="mt-1 block w-full bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                                           placeholder="Endpoint URL" value={endpoint}
                                           onChange={(e) => setEndpoint(e.target.value)}
                                           onBlur={(e) => validateEndpointAndUpdate(e.target.value)}/>
                                </div>
                            </div>
                        </div>
                        <div className="flex items-center mb-6 w-full">
                            <div className="flex items-center w-1/2">
                                <label className="block text-sm font-medium text-gray-300 w-1/2 px-3">
                                    Model Selection
                                    <SettingsTooltip tooltipIndex={3} tooltipVisible={() => tooltipVisible}
                                                     setTooltipVisible={setTooltipVisible}>
                                        Select the AI model for speech synthesis.
                                    </SettingsTooltip>
                                </label>
                                <select
                                    value={currentVoiceConfig.model}
                                    onChange={(e) => handleModelSelectionChange(e.target.value)}
                                    className="block w-1/2 bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                                >
                                    {modelOptions.map((option) => (
                                        <option key={option.value} value={option.value}>
                                            {option.name}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <div className="flex items-center w-1/2">
                                <label className="block text-sm font-medium text-gray-300 w-1/2 px-3">
                                    Operation Mode
                                    <SettingsTooltip tooltipIndex={4} tooltipVisible={() => tooltipVisible}
                                                     setTooltipVisible={setTooltipVisible}>
                                        Different models may support different operation modes, like voice cloning,
                                        Single-Speaker-TTS, Realtime Speech-To-Speech etc.
                                    </SettingsTooltip>
                                </label>
                                <select
                                    value={currentVoiceConfig.operation_mode}
                                    onChange={(e) => handleOperationModeChange(e.target.value)}
                                    className="block w-1/2 bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                                >
                                    {modelOperationModes[currentVoiceConfig.model] ? (
                                        modelOperationModes[currentVoiceConfig.model].map((option) => (
                                            <option key={option.value} value={option.value}>
                                                {option.name}
                                            </option>
                                        ))
                                    ) : (
                                        <option key="" value="">Default</option>
                                    )}
                                </select>
                            </div>
                        </div>
                        {currentVoiceConfig.operation_mode === "voice_cloning" && (
                            <div className="w-full">
                                <div className="flex items-center mb-2 w-full">
                                    <h2 className="text-l font-bold text-gray-300">
                                        Voice Embedding Settings
                                        <SettingsTooltip tooltipIndex={5} tooltipVisible={() => tooltipVisible}
                                                         setTooltipVisible={setTooltipVisible}>
                                            <span className="font-medium">
                                                A voice embedding stores the vocal characteristics of a speaker. AI
                                                Speech Frameworks can use this vocal data to align their output when performing
                                                voice cloning.
                                                <br/>
                                                <br/><span className="text-orange-400">CAUTION: Embeddings of different models are usually not compatible with each other.</span>
                                            </span>
                                        </SettingsTooltip>
                                    </h2>
                                </div>
                                <div className="flex items-center mb-6 w-full">
                                    <div className="flex items-center mt-2 w-2/3">
                                        <label className="block text-sm font-medium text-gray-300 w-1/3 px-3">
                                            Voice File
                                            <SettingsTooltip tooltipIndex={6} tooltipVisible={() => tooltipVisible}
                                                             setTooltipVisible={setTooltipVisible}>
                                                Select a local audio file to generate a voice embedding from.
                                            </SettingsTooltip>
                                        </label>
                                        <div className="w-2/3 px-3">
                                            <input
                                                type="file"
                                                accept=".wav,.mp3,.flac"
                                                onChange={handleEmbeddingFileChange}
                                                className="block w-full text-sm text-orange-400 file:mr-4 file:py-1 file:px-4 file:border-0 file:text-sm file:font-semibold file:bg-neutral-700 file:text-orange-400 hover:file:bg-neutral-500 file:cursor-pointer"
                                            />
                                        </div>
                                    </div>
                                    <div className="flex items-center mt-2 w-1/3">
                                        <div className="w-full">
                                            <HarmonyAudioPlayer className="w-full" src={embeddingFileAudio}/>
                                        </div>
                                    </div>
                                </div>
                                <div className="flex items-center mb-6 w-full">
                                    <div className="flex items-center w-2/3 px-3">
                                        <div className="flex items-center w-1/3">
                                        </div>
                                        <div className="flex items-center w-2/3 px-1">
                                            <div className="w-full flex items-center">
                                                <button
                                                    onClick={handleGenerateEmbedding}
                                                    className="bg-neutral-700 hover:bg-neutral-500 font-semibold py-1 px-2 mx-1 text-sm text-orange-400"
                                                >
                                                    Generate Embedding
                                                </button>
                                                <label className="text-sm font-medium text-gray-300">
                                                    <SettingsTooltip tooltipIndex={7} tooltipVisible={() => tooltipVisible}
                                                                     setTooltipVisible={setTooltipVisible}>
                                                        This sends an embedding Request for the provided audio file to the
                                                        endpoint provided.
                                                        <br/>The received embedding data will be stored in the current voice
                                                        configuration.
                                                        <br/>
                                                        <br/><span className="text-orange-400">CAUTION: Existing embedding data will be replaced.</span>
                                                    </SettingsTooltip>
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex items-center w-1/3 px-3">
                                        <div className="w-full flex items-center">
                                            <div className="w-1/2 px-3 flex items-center">
                                                {currentVoiceConfig && currentVoiceConfig.target_embedding &&
                                                    <div style={{width: '50px', height: '50px'}}>
                                                        <Heatmap data={currentVoiceConfig.target_embedding}
                                                                 colorRange={[0, 0.3]}/>
                                                    </div>
                                                }
                                            </div>
                                            <div className="w-1/2 px-3 flex items-center">
                                                <span className="text-sm text-orange-400">{embeddingStatus}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                        <div className="flex items-center mb-2 w-full">
                            <h2 className="text-l font-bold text-gray-300">
                                Voice Generation Settings
                                <SettingsTooltip tooltipIndex={8} tooltipVisible={() => tooltipVisible}
                                                 setTooltipVisible={setTooltipVisible}>
                                            <span className="font-medium">
                                                Settings for the model when generating speech.
                                                <br/>
                                                <br/>ATTENTION: Not all models support all settings. Please refer to the <span
                                                className="text-orange-400"><a
                                                href="https://github.com/harmony-ai-solutions/harmony-speech-engine/blob/main/docs/models.md"
                                                target="_blank">Documentation</a></span> for possible settings.
                                            </span>
                                </SettingsTooltip>
                            </h2>
                        </div>
                        <div className="flex items-center mb-6 w-1/2">
                            <label className="block text-sm font-medium text-gray-300 w-1/2 px-3">
                                Language
                                <SettingsTooltip tooltipIndex={9} tooltipVisible={() => tooltipVisible}
                                                 setTooltipVisible={setTooltipVisible}>
                                    Select the target language for voice synthesis.
                                    <br/>Not all models support multiple languages.
                                </SettingsTooltip>
                            </label>
                            <select
                                value={currentVoiceConfig.language}
                                onChange={(e) => handleLanguageSelectionChange(e.target.value)}
                                className="block w-1/2 bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                            >
                                {modelLanguageOptions[currentVoiceConfig.model] ? (
                                    modelLanguageOptions[currentVoiceConfig.model].map((option) => (
                                        <option key={option.value} value={option.value}>
                                            {option.name}
                                        </option>
                                    ))
                                ) : (
                                    <option key="" value="">Default</option>
                                )}
                            </select>
                        </div>
                        <div className="flex items-center mb-6 w-1/2">
                            <label className="block text-sm font-medium text-gray-300 w-1/2 px-3">
                                Voice
                                <SettingsTooltip tooltipIndex={10} tooltipVisible={() => tooltipVisible}
                                                 setTooltipVisible={setTooltipVisible}>
                                    Select the target voice for voice synthesis.
                                    <br/>Not all models support output voices.
                                </SettingsTooltip>
                            </label>
                            <select
                                value={currentVoiceConfig.voice}
                                onChange={(e) => handleVoiceSelectionChange(e.target.value)}
                                className="block w-1/2 bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                            >
                                {modelVoiceOptions[currentVoiceConfig.model] && modelVoiceOptions[currentVoiceConfig.model][currentVoiceConfig.language] ? (
                                    modelVoiceOptions[currentVoiceConfig.model][currentVoiceConfig.language].map((option) => (
                                        <option key={option.value} value={option.value}>
                                            {option.name}
                                        </option>
                                    ))
                                ) : (
                                    <option key="" value="">Default</option>
                                )}
                            </select>
                        </div>
                        <div className="flex items-center mb-6 w-1/2">
                            <label className="block text-sm font-medium text-gray-300 w-1/2 px-3">
                                Seed
                            </label>
                            <div className="w-1/2">
                                <input
                                    type="number"
                                    min="0"
                                    step="1"
                                    name="seed"
                                    className="mt-1 block w-full bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                                    placeholder="Seed"
                                    value={currentVoiceConfig.seed}
                                    onChange={(e) =>
                                        setCurrentVoiceConfig({
                                            ...currentVoiceConfig,
                                            seed: parseInt(e.target.value)
                                        })
                                    }
                                />
                            </div>
                        </div>
                        <div className="flex items-center mb-6 w-1/2">
                            <label className="block text-sm font-medium text-gray-300 w-1/2 px-3">
                                Style
                            </label>
                            <div className="w-1/2">
                                <input
                                    type="number"
                                    min="0"
                                    step="1"
                                    name="style"
                                    className="mt-1 block w-full bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                                    placeholder="Style"
                                    value={currentVoiceConfig.style}
                                    onChange={(e) =>
                                        setCurrentVoiceConfig({
                                            ...currentVoiceConfig,
                                            style: parseInt(e.target.value)
                                        })
                                    }
                                />
                            </div>
                        </div>
                        <div className="flex items-center mb-6 w-1/2">
                            <label className="block text-sm font-medium text-gray-300 w-1/2 px-3">
                                Speed
                            </label>
                            <div className="w-1/2">
                                <input
                                    type="number"
                                    step="0.01"
                                    name="speed"
                                    className="mt-1 block w-full bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                                    placeholder="Speed"
                                    value={currentVoiceConfig.speed}
                                    onChange={(e) =>
                                        setCurrentVoiceConfig({
                                            ...currentVoiceConfig,
                                            speed: parseFloat(e.target.value)
                                        })
                                    }
                                />
                            </div>
                        </div>
                        <div className="flex items-center mb-6 w-1/2">
                            <label className="block text-sm font-medium text-gray-300 w-1/2 px-3">
                                Pitch
                            </label>
                            <div className="w-1/2">
                                <input
                                    type="number"
                                    step="0.01"
                                    name="pitch"
                                    className="mt-1 block w-full bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                                    placeholder="Pitch"
                                    value={currentVoiceConfig.pitch}
                                    onChange={(e) =>
                                        setCurrentVoiceConfig({
                                            ...currentVoiceConfig,
                                            pitch: parseFloat(e.target.value)
                                        })
                                    }
                                />
                            </div>
                        </div>
                        <div className="flex items-center mb-6 w-1/2">
                            <label className="block text-sm font-medium text-gray-300 w-1/2 px-3">
                                Energy
                            </label>
                            <div className="w-1/2">
                                <input
                                    type="number"
                                    step="0.01"
                                    name="energy"
                                    className="mt-1 block w-full bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                                    placeholder="Energy"
                                    value={currentVoiceConfig.energy}
                                    onChange={(e) =>
                                        setCurrentVoiceConfig({
                                            ...currentVoiceConfig,
                                            energy: parseFloat(e.target.value)
                                        })
                                    }
                                />
                            </div>
                        </div>
                        <div className="flex items-center mb-2 w-full">
                            <h2 className="text-l font-bold text-gray-300">
                                Generate Speech
                                <SettingsTooltip tooltipIndex={11} tooltipVisible={() => tooltipVisible}
                                                 setTooltipVisible={setTooltipVisible}>
                                            <span className="font-medium">
                                                Here you can test above configuration for the speech engine by generating
                                                sample speech audio.
                                            </span>
                                </SettingsTooltip>
                            </h2>
                        </div>
                        <div className="flex items-center mb-2 w-full">
                            <div className="flex items-center w-2/3">
                                <label className="block text-sm font-medium text-gray-300 w-1/4 px-3">
                                    Input Text
                                    <SettingsTooltip tooltipIndex={12} tooltipVisible={() => tooltipVisible}
                                                     setTooltipVisible={setTooltipVisible}>
                                        Enter the text you want to convert to speech.
                                    </SettingsTooltip>
                                </label>
                                <div className="w-3/4 px-3">
                                    <textarea name="generation_text"
                                              className="mt-1 block w-full bg-neutral-800 min-h-24 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                                              placeholder="Type text to generate speech from here"
                                              value={generationText}
                                              onChange={(e) => setGenerationText(e.target.value)}
                                              onBlur={(e) => setGenerationText(e.target.value)}/>
                                </div>
                            </div>
                            <div className="flex items-center w-1/3">
                                <div className="flex flex-wrap items-center w-full">
                                    <div className="w-full mb-2">
                                        <button
                                            onClick={handleSynthesizeVoice}
                                            className="bg-neutral-700 hover:bg-neutral-500 font-semibold py-1 px-2 text-sm text-orange-400 w-full"
                                        >
                                            Generate Speech
                                        </button>
                                    </div>
                                    <div className="w-full">
                                        <HarmonyAudioPlayer className="w-full" src={generatedAudio}/>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {isModalVisible && (
                <div className="fixed inset-0 bg-gray-600/50">
                    <div
                        className="relative top-10 mx-auto p-5 border border-neutral-800 w-96 shadow-lg rounded-md bg-neutral-900">
                        <div className="mt-3 text-center">
                            <div
                                className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-200">
                                <svg className="h-6 w-6 text-red-600" fill="none" viewBox="0 0 24 24"
                                     stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                                          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                                </svg>
                            </div>
                            <h3 className="text-lg leading-6 font-medium text-orange-500 mt-4">{modalTitle}</h3>
                            <div className="mt-2 px-7 py-3">
                                <p className="text-sm text-gray-200">{modalMessage}</p>
                            </div>
                            <div className="items-center px-4 py-3">
                                <button onClick={() => setIsModalVisible(false)}
                                        className="px-4 py-2 bg-gray-500 text-white text-base font-medium rounded-md w-full shadow-sm hover:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-300">
                                    Close
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
            {confirmModalVisible && (
                <div className="fixed inset-0 bg-gray-600/50">
                    <div className="relative top-10 mx-auto p-5 border border-neutral-800 w-96 shadow-lg rounded-md bg-neutral-900">
                        <div className="mt-3 text-center">
                            <h3 className="text-lg leading-6 font-medium text-orange-500">{confirmModalTitle}</h3>
                            <div className="mt-2 px-7 py-3">
                                <p className="text-sm text-gray-200">{confirmModalMessage}</p>
                            </div>
                            {confirmModalHasInput && <div className="mt-2 px-7 py-3">
                                <input type="text" name="confirm_modal_input"
                                       className="mt-1 block w-full bg-neutral-800 shadow-sm focus:outline-none focus:border-orange-400 border border-neutral-600 text-neutral-100"
                                       placeholder={confirmModalMessage} value={confirmModalInput}
                                       onChange={(e) => setConfirmModalInput(e.target.value)}
                                       onBlur={(e) => setConfirmModalInput(e.target.value)}/>
                            </div>}
                            <div className="flex justify-center gap-4 pt-3">
                                <button onClick={() => {
                                    setConfirmModalVisible(false);
                                    if (confirmModalHasInput) {
                                        confirmModalYes(confirmModalInput);
                                    } else {
                                        confirmModalYes(null);
                                    }
                                }}
                                        className="bg-neutral-700 hover:bg-neutral-500 font-bold py-1 px-2 mx-1 text-orange-400">
                                    Confirm
                                </button>
                                <button onClick={() => {
                                    setConfirmModalVisible(false);
                                    confirmModalNo(); }}
                                        className="bg-red-700 hover:bg-red-500 font-bold py-1 px-2 mx-1 text-white">
                                    Abort
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}

export default TTS;