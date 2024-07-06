# Project Harmony.AI - Harmony Speech Engine

---

Harmony Speech Engine is a high performance Engine for Inference of Open Source Speech AI.
It's designed to serve as the backbone for providing Project Harmony.AI's Speech Platform "Harmony Speech".

#### - Currently under Development -

## Goals & Features
The goal of this engine is to provide a reliable and easy-to-maintain service which can be used for deploying Open Source
AI Speech technologies. Each of these Technologies have different setup requirements and pre-conditions, so the goal of
this project is to unify these requirements in a way that these technologies can work together seamlessly.

Aside from providing a runtime for these technologies behind a unified service API, the Harmony Speech Engine also
allows for recombining different technologies on-the-fly, to reduce processing duration and latency. For example, you
can generate Speech using a TTS integration, and then apply additional filtering using voice conversion.

The rough Idea for this project is to become something like [vLLM](https://github.com/vllm-project/vllm) / [Aphrodite Engine](https://github.com/PygmalionAI/Aphrodite-engine)
for AI Speech Inference. Significant parts of the codebase have been forked from Aphrodite engine, with a couple
modifications to allow for the intended use case.
Support and Ideas for Improving this Project are very welcome.

### Differences from forked [Aphrodite Engine](https://github.com/PygmalionAI/Aphrodite-engine)
- Per-Request processing, instead of token sequence batching
- Support for loading and executing multiple models in parallel
- No *distributed* execution of single models (i.e. sharding); mainly to reduce complexity
- No general quantization; if quantization is supported, this will be part of the individual model config.

### Planned and availiable Integrations
The following Technologies and Features are planned to be supported

- [ ] Zero-Shot Voice Conversion
  - [ ] NaturalSpeech3 Voice Converter
  - [ ] OpenVoice Tone Converter
- [ ] Multi-Shot Voice Conversion
  - [ ] StyleTTS 2 Voice Converter
  - [ ] RVC (Retrieval-base-Voice-Conversion)
- [ ] Multi-Shot Voice Cloning
  - [ ] StyleTTS 2
- [ ] Zero-Shot Voice Cloning
  - [ ] Harmony Speech V1 (english)
  - [ ] Vall-E-X (Multilingual)
- [ ] Generic Single-Speaker TTS
  - [ ] OpenVoice V1 (English / Chinese + basic emotions)
  - [ ] OpenVoice V2 (English, Spanish, French, Chinese, Japanese and Korean)
- [ ] Generic Multispeaker TTS
  - [ ] EmotiVoice (English / Chinese + basic emotions for a wide range of speakers)
- [ ] Adaptive Voice Cloning
  - [ ] Basic Overlays (TTS + Voice Conversion)
  - [ ] Embedding Vector Matching (Convenience Feature)
- [ ] Automatic Speech Recognition & Language Detection
  - [ ] Whisper
  - [ ] Distil-Whisper


---

## About Project Harmony.AI
![Project Harmony.AI](docs/images/Harmony-Main-Banner-200px.png)
### Our goal: Elevating Human <-to-> AI Interaction beyond known boundaries.
Project Harmony.AI emerged from the idea to allow for a seamless living together between AI-driven characters and humans.
Since it became obvious that a lot of technologies required for achieving this goal are not existing or still very experimental,
the long term vision of Project Harmony is to establish the full set of technologies which help minimizing biological and
technological barriers in Human <-to-> AI Interaction.

### Our principles: Fair use and accessibility
We want to counter today's tendencies of AI development centralization at the hands of big
corporations. We're pushing towards maximum transparency in our own development efforts, and aim for our software to be
accessible and usable in the most democratic ways possible.

Therefore, for all our current and future software offerings, we'll perform a constant and well-educated evaluation whether
we can safely open source them in parts or even completely, as long as this appears to be non-harmful towards achieving
the project's main goal.

Harmony Speech Engine is being distributed under the AGPLv3 License, because A lot of the code in the module `harmonyspeech` has been borrowed from [Aphrodite Engine](https://github.com/PygmalionAI/Aphrodite-engine).
Everyone can use this software as part of their own projects without any restrictions from our side, except from restrictions derived from the nature of the licensing.

### How to reach out to us

[Official Website of Project Harmony.AI](https://project-harmony.ai/)

#### If you want to collaborate or support this Project financially:

Feel free to join our Discord Server and / or subscribe to our Patreon - Even $1 helps us drive this project forward.

![Harmony.AI Discord Server](docs/images/discord32.png) [Harmony.AI Discord Server](https://discord.gg/f6RQyhNPX8)

![Harmony.AI Discord Server](docs/images/patreon32.png) [Harmony.AI Patreon](https://patreon.com/harmony_ai)

#### If you want to use our software commercially or discuss a business or development partnership:

Contact us directly via: [contact@project-harmony.ai](mailto:contact@project-harmony.ai)