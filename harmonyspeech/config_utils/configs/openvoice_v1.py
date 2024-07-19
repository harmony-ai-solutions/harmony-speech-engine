

class OpenVoiceV1SynthesizerConfig:

    def __init__(
        self,
        mel_n_channels: int = 40,
        model_hidden_size: int = 768,
        model_num_layers: int = 3,
        model_embedding_size: int = 768
    ):
        self.mel_n_channels = mel_n_channels
        self.model_hidden_size = model_hidden_size
        self.model_num_layers = model_num_layers
        self.model_embedding_size = model_embedding_size


class OpenVoiceV1ToneConverterConfig:

    def __init__(
        self,
        mel_n_channels: int = 40,
        model_hidden_size: int = 768,
        model_num_layers: int = 3,
        model_embedding_size: int = 768
    ):
        self.mel_n_channels = mel_n_channels
        self.model_hidden_size = model_hidden_size
        self.model_num_layers = model_num_layers
        self.model_embedding_size = model_embedding_size
