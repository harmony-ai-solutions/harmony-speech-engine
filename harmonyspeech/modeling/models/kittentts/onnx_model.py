import threading

from misaki import en, espeak  # noqa: F401

import numpy as np
import onnxruntime as ort
import phonemizer
import soundfile as sf

from harmonyspeech.modeling.models.kittentts.preprocess import TextPreprocessor

# KittenTTS graphs are **stochastic by construction** (verified 2026-11-09).
#
# The exported ONNX graphs carry ``RandomNormalLike`` / ``RandomUniformLike``
# nodes **without a ``seed`` attribute**, so ONNX Runtime draws fresh noise on
# every ``session.run``. Two runs of the same text are therefore never
# sample-identical (measured on ``kitten-tts-mini-0.8``: two fresh processes
# differed by max |Δsample| 0.14 with corr 0.992 and identical duration; the
# 6-chunk stitched comparison gave log-mel MSE ≈ 3.0 dB², corr 0.992).
# ONNX Runtime exposes no RNG seed, so the engine cannot make this
# deterministic: byte-identity gates must treat the family as sampling (see
# ``scripts/audio_regression.MODEL_SAMPLES_EVERY_RUN`` and the Phase 12-4 log).

#: Serialises espeak phonemisation (Phase 8-5).
#:
#: ``phonemizer``'s ``EspeakBackend`` keeps mutable post-processing state
#: (``WordsMismatch`` buffers the last input) and is not thread-safe: two
#: concurrent ``phonemize`` calls on one backend fail with
#: ``RuntimeError: number of lines in input and output must be equal, we have:
#: input=1, output=2`` (reproduced on this machine while measuring the lanes).
#: Lane execution therefore holds this lock around every phonemise call. One
#: call is ~0.3 ms, so the serialisation costs nothing next to a ~1.5 s
#: ``session.run``.
PHONEMIZER_LOCK = threading.Lock()

#: Fixed ORT intra-op thread count for KittenTTS sessions (Phase 8-5).
#:
#: The engine runs one request per lane (``max_batch_size`` lanes) on one shared
#: session, so the worry was that ORT's default "one intra-op thread per physical
#: core" oversubscribes the CPU by the lane count. The measured matrix
#: (lanes x intra_op, mini variant, 12-core dev machine, ``docs/batching.md``)
#: refuted it: 8 lanes run at 1.22 / 1.24 / 1.27 / 1.27 req/s for intra_op
#: 1 / 2 / 4 / default, i.e. the pin costs throughput and buys nothing, and
#: intra_op=1 also doubles the single-request latency (2.61 s vs 1.61 s for
#: 5.9 s of audio). ``0`` means ORT's own default, i.e. one intra-op thread per
#: physical core — which also scales down on small hosts, the deployment target
#: of this model. It is set **explicitly** so the lane behaviour is pinned
#: rather than inherited from a future ORT default, and it is deliberately not
#: configurable.
KITTEN_TTS_INTRA_OP_NUM_THREADS = 0


def build_session_options(intra_op_num_threads=None) -> ort.SessionOptions:
    """Return the ORT session options for a KittenTTS session.

    ``intra_op_num_threads`` defaults to :data:`KITTEN_TTS_INTRA_OP_NUM_THREADS`;
    an explicit value is only used by the Phase 8-5 measurement matrix.
    """
    if intra_op_num_threads is None:
        intra_op_num_threads = KITTEN_TTS_INTRA_OP_NUM_THREADS
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(intra_op_num_threads)
    # One lane == one request; the lanes provide the parallelism across requests.
    options.inter_op_num_threads = 1
    return options


def basic_english_tokenize(text):
    """Basic English tokenizer that splits on whitespace and punctuation."""
    import re

    tokens = re.findall(r"\w+|[^\w\s]", text)
    return tokens


def ensure_punctuation(text):
    """Ensure text ends with punctuation. If not, add a comma."""
    text = text.strip()
    if not text:
        return text
    if text[-1] not in ".!?,;:":
        text = text + ","
    return text


def chunk_text(text, max_len=400):
    """Split text into chunks for processing long texts."""
    import re

    sentences = re.split(r"[.!?]+", text)
    chunks = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) <= max_len:
            chunks.append(ensure_punctuation(sentence))
        else:
            # Split long sentences by words
            words = sentence.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_len:
                    temp_chunk += " " + word if temp_chunk else word
                else:
                    if temp_chunk:
                        chunks.append(ensure_punctuation(temp_chunk.strip()))
                    temp_chunk = word
            if temp_chunk:
                chunks.append(ensure_punctuation(temp_chunk.strip()))

    return chunks


class TextCleaner:
    def __init__(self, dummy=None):
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

        dicts = {}
        for i in range(len(symbols)):
            dicts[symbols[i]] = i

        self.word_index_dictionary = dicts

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
        return indexes


class KittenTTS_1_Onnx:
    def __init__(
        self, model_path="kitten_tts_nano_preview.onnx", voices_path="voices.npz", speed_priors={}, voice_aliases={}
    ):
        """Initialize KittenTTS with model and voice data.

        Args:
            model_path: Path to the ONNX model file
            voices_path: Path to the voices NPZ file
        """
        self.model_path = model_path
        self.voices = np.load(voices_path)
        # One shared session for all lanes: ORT ``InferenceSession.run`` is
        # thread-safe and releases the GIL, so lane threads really do run in
        # parallel on N cores without N copies of the weights (Phase 8-5).
        self.session = ort.InferenceSession(model_path, sess_options=build_session_options())

        self.phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True
        )
        self.text_cleaner = TextCleaner()
        self.speed_priors = speed_priors

        # Available voices
        self.available_voices = [
            "expr-voice-2-m",
            "expr-voice-2-f",
            "expr-voice-3-m",
            "expr-voice-3-f",
            "expr-voice-4-m",
            "expr-voice-4-f",
            "expr-voice-5-m",
            "expr-voice-5-f",
        ]
        self.all_voice_names = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
        self.voice_aliases = voice_aliases

        self.preprocessor = TextPreprocessor(remove_punctuation=False)

    def _prepare_inputs(self, text: str, voice: str, speed: float = 1.0) -> dict:
        """Prepare ONNX model inputs from text and voice parameters."""
        if voice in self.voice_aliases:
            voice = self.voice_aliases[voice]

        if voice not in self.available_voices:
            raise ValueError(f"Voice '{voice}' not available. Choose from: {self.available_voices}")

        if voice in self.speed_priors:
            speed = speed * self.speed_priors[voice]

        # Phonemize the input text. The espeak backend is not thread-safe, so
        # lane execution serialises this call (see PHONEMIZER_LOCK).
        with PHONEMIZER_LOCK:
            phonemes_list = self.phonemizer.phonemize([text])

        # Process phonemes to get token IDs
        phonemes = basic_english_tokenize(phonemes_list[0])
        phonemes = " ".join(phonemes)
        tokens = self.text_cleaner(phonemes)

        # Add start and end tokens
        tokens.insert(0, 0)
        tokens.append(10)
        tokens.append(0)

        input_ids = np.array([tokens], dtype=np.int64)
        ref_id = min(len(text), self.voices[voice].shape[0] - 1)
        ref_s = self.voices[voice][ref_id : ref_id + 1]

        return {"input_ids": input_ids, "style": ref_s, "speed": np.array([speed], dtype=np.float32)}

    def generate(
        self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0, clean_text: bool = True
    ) -> np.ndarray:
        out_chunks = []
        if clean_text:
            text = self.preprocessor(text)
        for text_chunk in chunk_text(text):
            out_chunks.append(self.generate_single_chunk(text_chunk, voice, speed))
        return np.concatenate(out_chunks, axis=-1)

    def generate_single_chunk(self, text: str, voice: str = "expr-voice-5-m", speed: float = 1.0) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)

        Returns:
            Audio data as numpy array
        """
        onnx_inputs = self._prepare_inputs(text, voice, speed)

        outputs = self.session.run(None, onnx_inputs)

        # Trim audio
        audio = outputs[0][..., :-5000]

        return audio

    def warmup(self) -> None:
        """Run one minimal generate so the first real request skips the cold start.

        The first ``session.run`` on a freshly loaded session pays graph
        preparation and arena allocation; the Phase 8-5 measurement recorded a
        ~2 s cold first generate against a ~1.5 s warm one. The dummy text is the
        shortest chunk the pipeline can produce, so the price is one small
        inference at load time. Failures are the caller's to log and ignore: a
        warmup must never turn a loadable model into a failed one.
        """
        self.generate("Hi", voice="expr-voice-5-m", speed=1.0)

    def generate_to_file(
        self,
        text: str,
        output_path: str,
        voice: str = "expr-voice-5-m",
        speed: float = 1.0,
        sample_rate: int = 24000,
        clean_text: bool = True,
    ) -> None:
        """Synthesize speech and save to file.

        Args:
            text: Input text to synthesize
            output_path: Path to save the audio file
            voice: Voice to use for synthesis
            speed: Speech speed (1.0 = normal)
            sample_rate: Audio sample rate
            clean_text: If true, it will cleanup the text. Eg. replace numbers with words.
        """
        audio = self.generate(text, voice, speed, clean_text=clean_text)
        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path}")
