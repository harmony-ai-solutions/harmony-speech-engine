# Standard voice pack — provenance

These 28 WAV clips are vendored verbatim from the **Chatterbox-TTS-Server**
project's `voices/` directory (MIT licensed; see the `LICENSE` file in this
directory for the full upstream license text, `Copyright (c) 2025 devnen`).

* Source: `Chatterbox-TTS-Server/voices/*.wav`
* Vendored: 2026-11-09, Phase 13-1 of the engine-modernization plan
* Format: 44.1 kHz, mono, 16-bit PCM WAV; 7.3 s – 9.7 s per clip
* Names: `Abigail`, `Adrian`, `Alexander`, `Alice`, `Austin`, `Axel`,
  `Connor`, `Cora`, `Elena`, `Eli`, `Emily`, `Everett`, `Gabriel`, `Gianna`,
  `Henry`, `Ian`, `Jade`, `Jeremiah`, `Jordan`, `Julian`, `Layla`, `Leonardo`,
  `Michael`, `Miles`, `Olivia`, `Ryan`, `Taylor`, `Thomas`

The files are the speaker identities: they are kept byte-for-byte, including
the upstream names (owner decision 2026-11-08). Each model family embeds them
in its own embedding space, so the same name means "the same reference clip",
not "identical rendered audio".

Do not rename or re-encode these files without updating the catalog test that
pins the name set — the names are the public voice identifiers.
