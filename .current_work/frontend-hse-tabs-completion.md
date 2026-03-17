# Plan: Complete HSE Frontend Tabs

## Overview

The Harmony Speech Engine frontend (`frontend/`) currently only has a fully functional TTS tab. The STT and VAD tabs exist but are missing testing sections, and Voice Conversion, Audio Restoration, and Settings tabs are entirely absent. This plan covers all required changes to complete the frontend UI.

## Current State

| Tab | Status | Missing |
|-----|--------|---------|
| Text-to-Speech | ✅ Complete | — |
| Speech-to-Text | ⚠️ Partial | Audio upload, transcription test, result display |
| Voice Activity Detection | ⚠️ Partial | Audio upload, detection test, result display |
| Voice Conversion | ❌ Missing | Entire tab |
| Audio Restoration | ❌ Missing | Entire tab |
| Settings | ❌ Missing | Entire tab |

**Root cause for Audio Restoration:** The `docs/api/openapi.json` spec does not include the `/v1/audio/convert` and `/v1/audio/convert/models` API endpoints, so the generated `@harmony-ai/harmonyspeech` npm package has no methods for them. This must be fixed as a prerequisite.

---

## Phase 0: Prerequisite — Automate OpenAPI Spec Extraction & NPM Package Update

The `docs/api/openapi.json` is currently maintained manually, but FastAPI automatically generates the complete OpenAPI spec at runtime via its `/openapi.json` endpoint. The `/v1/audio/convert` and `/v1/audio/convert/models` routes are already registered in [`api_server.py`](harmonyspeech/endpoints/openai/api_server.py) but absent from the committed spec because the spec was never re-fetched after those routes were added.

### How the JS Client Build Works (Current State)

The current local workflow (undocumented) is:
1. `docs/api/generate_openapi_js.sh` runs `openapitools/openapi-generator-cli` via Docker, reading `openapi.json` and writing the generated JS client into `docs/api/jsclient/`
2. The generated `jsclient/src/` directory contains: `api/DefaultApi.js`, `model/*.js`, `ApiClient.js`, `index.js`
3. **`HarmonySpeech.js`** — the hand-written `HarmonySpeechEnginePlugin` convenience wrapper — must be **manually copied into** `jsclient/src/` alongside the generated files (the generator does not create it)
4. `npm run build` inside `jsclient/` runs babel to compile `src/` → `dist/`
5. The `jsclient/` package is then published to npm as `@harmony-ai/harmonyspeech`

This process is currently **undocumented and entirely manual**. The tasks below fix this.

### 0.1 Create `docs/api/HarmonySpeech.js` Source File

**New file:** [`docs/api/HarmonySpeech.js`](docs/api/HarmonySpeech.js)

Move/copy the hand-written `HarmonySpeechEnginePlugin` wrapper source (currently only existing in the installed `dist/HarmonySpeech.js`) into `docs/api/` as the canonical source. This file will be copied into `jsclient/src/` by the build script.

**Add two new methods to this file:**

**`convertAudio(audioConversionRequest, options = {})`:**
- Validates `audioConversionRequest` is a non-null object
- Injects `this.apiKey` into `options.xApiKey` and `options.apiKey` (same pattern as all other methods)
- Calls `this.defaultApi.convertAudioV1AudioConvertPost(audioConversionRequest, options)`
- Same `try/catch` + `this.handleError('Error converting audio:', error)` pattern

**`showAvailableAudioConversionModels(options = {})`:**
- Injects `this.apiKey` into options
- Calls `this.defaultApi.showAvailableAudioConversionModelsV1AudioConvertModelsGet(options)`
- Same error handling pattern

### 0.2 Create OpenAPI Spec Extraction Script

**New file:** [`docs/api/generate_openapi_spec.sh`](docs/api/generate_openapi_spec.sh)

This script starts the HSE server with no models loaded (API routes are still registered), fetches the live spec from FastAPI's built-in `/openapi.json` endpoint, and saves it as `docs/api/openapi.json`:

```bash
#!/bin/bash
set -e

cd "$(dirname "$0")/../../"  # run from repo root

echo "Starting HSE server for OpenAPI spec generation..."
python -m harmonyspeech.endpoints.cli --host 0.0.0.0 --port 12080 &
SERVER_PID=$!

# Wait for server to be ready (up to 60s)
for i in {1..30}; do
  if curl -sf http://localhost:12080/health > /dev/null 2>&1; then
    echo "Server is ready."
    break
  fi
  echo "Waiting for server... ($i/30)"
  sleep 2
done

echo "Fetching OpenAPI spec..."
curl -sf http://localhost:12080/openapi.json -o docs/api/openapi.json
echo "Spec saved to docs/api/openapi.json"

echo "Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null || true
echo "Done."
```

### 0.3 Update `generate_openapi_js.sh` to Include Full Build Pipeline

**File:** [`docs/api/generate_openapi_js.sh`](docs/api/generate_openapi_js.sh)

Extend the existing script to be a complete end-to-end build pipeline after code generation:

```bash
# After existing openapi-generator-cli docker run call...

# Copy hand-written wrapper into generated source
echo "Copying HarmonySpeech.js wrapper..."
cp HarmonySpeech.js jsclient/src/HarmonySpeech.js

# Build (babel compile src -> dist)
echo "Building JS client..."
cd jsclient
npm install
npm run build
echo "Build complete. Package ready in jsclient/dist/"
```

Also add a `--version` parameter option to allow the CI to pass in a specific npm package version:
```bash
# At the top of the script, allow version override via environment variable
VERSION="${NPM_VERSION:-0.1.0a}"
# Then patch jsclient/package.json version before building
```

### 0.4 Create Documentation: `docs/api/README.md`

**New file:** [`docs/api/README.md`](docs/api/README.md)

Document the complete JS client generation and publication process so future contributors understand the workflow:

```markdown
# Harmony Speech Engine API Client Generation

## Overview
The JS client (`@harmony-ai/harmonyspeech`) is generated from the live FastAPI OpenAPI spec,
extended with a hand-written convenience wrapper, compiled via Babel, and published to npm.

## Scripts
- `generate_openapi_spec.sh` — starts the HSE server, fetches the live spec, saves to openapi.json
- `generate_openapi_js.sh` — runs openapi-generator-cli, copies HarmonySpeech.js, builds via babel

## Manual Process
1. Start with a fresh checkout and a working Python environment with HSE dependencies installed
2. Run `bash docs/api/generate_openapi_spec.sh` to fetch the live spec into docs/api/openapi.json
3. Run `bash docs/api/generate_openapi_js.sh` to regenerate and build the JS client
4. Run `cd docs/api/jsclient && npm publish --access public` to publish

## Automated CI Process
The npm publish is handled automatically as part of `docker-release-ui.yml` when a new tag is pushed.
The OpenAPI spec should be manually regenerated (step 2-3 above) whenever API routes change,
and the updated openapi.json committed to the repository.
```

### 0.5 Extend `docker-release-ui.yml` with npm Publish Job

**File:** [`.github/workflows/docker-release-ui.yml`](.github/workflows/docker-release-ui.yml)

Fold the npm client build and publish as a **prerequisite job** inside the existing `docker-release-ui.yml`. The existing Docker image build job will `needs: [publish-npm-client]`.

**Triggers:** Unchanged — tag-based (`on: push: tags: ['*']`) plus `workflow_dispatch`, matching the engine release pattern.

**New job: `publish-npm-client`** (added before the existing `build-and-push` job):

```yaml
publish-npm-client:
  runs-on: ubuntu-latest

  steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '20'
        registry-url: 'https://registry.npmjs.org'

    - name: Set up Docker Buildx (for openapi-generator-cli)
      uses: docker/setup-buildx-action@v2

    - name: Determine npm package version
      shell: bash
      run: |
        if [[ "$GITHUB_REF" == refs/tags/* ]]; then
          NPM_VERSION="${GITHUB_REF#refs/tags/v}"
        elif [[ -n "${{ github.event.inputs.version }}" ]]; then
          NPM_VERSION="${{ github.event.inputs.version }}"
        else
          NPM_VERSION="0.0.0-dev"
        fi
        echo "NPM_VERSION=$NPM_VERSION" >> $GITHUB_ENV

    - name: Regenerate and build JS client from committed openapi.json
      run: |
        cd docs/api
        NPM_VERSION="${{ env.NPM_VERSION }}" bash generate_openapi_js.sh

    - name: Publish @harmony-ai/harmonyspeech to npm
      run: cd docs/api/jsclient && npm publish --access public
      env:
        NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

**Key design decisions:**

- **Tag drives version** — npm package version is derived from the git tag (stripping `v` prefix): tag `v0.1.2` → npm `0.1.2`. Matches engine Docker image versioning.
- **`workflow_dispatch` reuses existing `version` input** — the existing `docker-release-ui.yml` already has a `version` input for `workflow_dispatch`, so no new input parameter is needed.

**Update existing `build-and-push` job** to depend on the new job:
```yaml
build-and-push:
  runs-on: ubuntu-latest
  needs: [publish-npm-client]   # <-- add this line
  # ... rest unchanged ...
```

The frontend Docker build (`Dockerfile.speech-engine`) runs `npm install` which will pull the freshly published `@harmony-ai/harmonyspeech` version. Update `frontend/package.json` in the `harmony-link-ui` repository to use the new version (or `"latest"` if the UI repo follows a floating dependency strategy).

### 0.6 Update Frontend `package.json`

**File:** [`frontend/package.json`](frontend/package.json)

After publishing the new npm package version, bump `@harmony-ai/harmonyspeech` to the new version. This can also be part of the CI workflow for docker releases — the `docker-release-ui.yml` could be extended to update `package.json` before building the Docker image.

---

## Phase 1: Extend STT Tab with Testing UI

### Files Modified
- [`frontend/src/components/modules/STTHarmonySpeechSettingsView.jsx`](frontend/src/components/modules/STTHarmonySpeechSettingsView.jsx)

### Changes

Add the following state variables:
```jsx
const [transcriptionFile, setTranscriptionFile] = useState(null);
const [transcriptionFileAudio, setTranscriptionFileAudio] = useState(null);
const [transcriptionResult, setTranscriptionResult] = useState('');
const [getLanguage, setGetLanguage] = useState(false);
const [getTimestamps, setGetTimestamps] = useState(false);
```

Add handler functions:
- `handleTranscriptionFileChange(e)` — reads file, stores as DataURL for preview in `HarmonyAudioPlayer`
- `handleTranscribe()` — reads file as base64, calls `harmonySpeechPlugin.createTranscription({ model, input_audio: base64, get_language: getLanguage, get_timestamps: getTimestamps })`, stores result text

Add a "Transcription Testing" section below the existing model selector row in the JSX:
```
<h2>Transcription Test</h2>
[Audio File Upload] + [HarmonyAudioPlayer preview]
[Checkbox: Return language] [Checkbox: Return timestamps]
[Transcribe button]
[Read-only textarea showing transcription result]
```

Import `HarmonyAudioPlayer` from `../widgets/HarmonyAudioPlayer.jsx`.

---

## Phase 2: Extend VAD Tab with Testing UI

### Files Modified
- [`frontend/src/components/modules/VADHarmonySpeechSettingsView.jsx`](frontend/src/components/modules/VADHarmonySpeechSettingsView.jsx)

### Changes

Add the following state variables:
```jsx
const [vadFile, setVadFile] = useState(null);
const [vadFileAudio, setVadFileAudio] = useState(null);
const [vadResult, setVadResult] = useState(null);
```

Add handler functions:
- `handleVadFileChange(e)` — reads file, stores as DataURL for preview
- `handleDetectVoiceActivity()` — reads file as base64, calls `harmonySpeechPlugin.detectVoiceActivity({ model, input_audio: base64 })`, stores result

Add a "VAD Testing" section below the existing model selector in the JSX:
```
<h2>Voice Activity Detection Test</h2>
[Audio File Upload] + [HarmonyAudioPlayer preview]
[Detect Voice Activity button]
[Result display: formatted JSON or human-readable activity timestamps]
```

VAD result display: render as a `<pre>` or formatted table showing speech segment start/end times and confidence values (from the `DetectVoiceActivityResponse` structure).

Import `HarmonyAudioPlayer` from `../widgets/HarmonyAudioPlayer.jsx`.

---

## Phase 3: New Voice Conversion Tab

### Files Created
- [`frontend/src/components/modules/VCHarmonySpeechSettingsView.jsx`](frontend/src/components/modules/VCHarmonySpeechSettingsView.jsx)

### Design

Modeled closely after `TTSHarmonySpeechSettingsView.jsx` but for voice conversion.

**Known model names map:**
```js
const knownModelNames = {
  "openvoice_v1": "OpenVoice V1",
  "openvoice_v2": "OpenVoice V2",
};
```

**State:**
```jsx
const [endpoint, setEndpoint] = useState(mergedSettings.endpoint);
const [model, setModel] = useState(mergedSettings.model);
const [modelOptions, setModelOptions] = useState([{name: "Error: no models available", value: null}]);
// Source audio
const [sourceFile, setSourceFile] = useState(null);
const [sourceFileAudio, setSourceFileAudio] = useState(null);
// Reference/target (for voice style transfer)
const [targetFile, setTargetFile] = useState(null);
const [targetFileAudio, setTargetFileAudio] = useState(null);
const [targetEmbedding, setTargetEmbedding] = useState('');
const [embeddingStatus, setEmbeddingStatus] = useState('No embedding loaded.');
// Generation options
const [speed, setSpeed] = useState(1.0);
const [pitch, setPitch] = useState(1.0);
const [energy, setEnergy] = useState(1.0);
// Output
const [resultAudio, setResultAudio] = useState(null);
// Plugin
const [harmonySpeechPlugin, setHarmonySpeechPlugin] = useState(null);
// Validation
const [validationState, setValidationState] = useState({ status: 'idle', message: '' });
```

**Sections:**
1. `ConfigVerificationSection` + endpoint URL input (same pattern as STT/VAD)
2. **Model Selection** — dropdown populated via `harmonySpeechPlugin.showAvailableVoiceConversionModels()`
3. **Source Audio** — file upload + `HarmonyAudioPlayer` preview
4. **Target Reference Audio** (voice style) — file upload + `HarmonyAudioPlayer` preview; "Generate Embedding" button (calls `harmonySpeechPlugin.createEmbedding({model, input_audio: base64})`); embedding status display with `Heatmap` preview
5. **Generation Options** — Speed, Pitch, Energy number inputs
6. **Convert Voice** button → calls `harmonySpeechPlugin.convertVoice({model, source_audio: base64, target_embedding, generation_options: {speed, pitch, energy}})` → result in `HarmonyAudioPlayer`

**Props interface:**
```jsx
const VCHarmonySpeechSettingsView = ({initialSettings, saveSettingsFunc}) => { ... }
```

**Imports:** `HarmonyAudioPlayer`, `Heatmap`, `SettingsTooltip`, `ConfigVerificationSection`, `ErrorDialog`, `ThemedSelect`, `HarmonySpeechEnginePlugin`, `isHarmonyLinkMode`, `mergeConfigWithDefaults`, `MODULE_DEFAULTS`

---

## Phase 4: New Audio Restoration Tab

### Files Created
- [`frontend/src/components/modules/AudioRestorationHarmonySpeechSettingsView.jsx`](frontend/src/components/modules/AudioRestorationHarmonySpeechSettingsView.jsx)

### Design

Similar to Voice Conversion but simpler (no target reference, no embedding).

**Known model names map:**
```js
const knownModelNames = {
  "voicefixer": "VoiceFixer",
};
```

**State:**
```jsx
const [endpoint, setEndpoint] = useState(mergedSettings.endpoint);
const [model, setModel] = useState(mergedSettings.model);
const [modelOptions, setModelOptions] = useState([{name: "Error: no models available", value: null}]);
const [sourceFile, setSourceFile] = useState(null);
const [sourceFileAudio, setSourceFileAudio] = useState(null);
const [resultAudio, setResultAudio] = useState(null);
const [harmonySpeechPlugin, setHarmonySpeechPlugin] = useState(null);
const [validationState, setValidationState] = useState({ status: 'idle', message: '' });
```

**Sections:**
1. `ConfigVerificationSection` + endpoint URL input
2. **Model Selection** — dropdown populated via `harmonySpeechPlugin.showAvailableAudioConversionModels()` (new method from Phase 0)
3. **Source Audio** — file upload + `HarmonyAudioPlayer` preview
4. **Restore Audio** button → calls `harmonySpeechPlugin.convertAudio({model, source_audio: base64})` (new method from Phase 0) → result in `HarmonyAudioPlayer`

**Props interface:**
```jsx
const AudioRestorationHarmonySpeechSettingsView = ({initialSettings, saveSettingsFunc}) => { ... }
```

---

## Phase 5: New Settings Tab

### Files Created
- [`frontend/src/components/SpeechEngineSettingsView.jsx`](frontend/src/components/SpeechEngineSettingsView.jsx)

### Design

A simplified version of [`GeneralSettingsView.jsx`](frontend/src/components/GeneralSettingsView.jsx) containing **only** the theme selector section. All Harmony Link-specific settings (working dir, database, API keys, ports, etc.) are omitted.

```jsx
import { useTheme } from '../contexts/ThemeContext';
import { listThemes } from '../services/management/themeService';

const SpeechEngineSettingsView = () => {
  const { currentTheme, switchTheme } = useTheme();
  const [themes, setThemes] = useState([]);

  useEffect(() => {
    listThemes().then(setThemes).catch(err => console.error("Failed to load themes:", err));
  }, []);

  return (
    <div className="flex flex-col min-h-full bg-background-base">
      <div className="bg-background-surface/30 ... px-6 py-4">
        <h1><span className="text-gradient-primary">Appearance</span> Settings</h1>
        <p>Theme and personalization settings</p>
      </div>
      <div className="flex-1 p-6 space-y-8 max-w-7xl">
        <section>
          <h2>Appearance & Personalization</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {themes.map((theme) => (
              <div key={theme.id} onClick={() => switchTheme(theme.id)} className={...}>
                {/* Theme preview card — identical to GeneralSettingsView lines 378-411 */}
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
```

No "Save Settings" button needed — theme switching is instant via `switchTheme()`.

---

## Phase 6: localStorage Fallback for Theme Service in Speech Engine Mode

### Files Modified
- [`frontend/src/services/management/themeService.js`](frontend/src/services/management/themeService.js)

### Design

The current theme service calls the Harmony Link management API (`/api/themes`, `/api/settings/current-theme`). In Speech Engine mode those APIs don't exist.

**Strategy:** Add mode-aware branching using `isHarmonySpeechEngineMode()` from [`appMode.js`](frontend/src/config/appMode.js).

**`getCurrentTheme()` changes:**
```js
export async function getCurrentTheme() {
  if (isHarmonySpeechEngineMode()) {
    const themeId = localStorage.getItem('hse-theme') || 'midnight-rose';
    const themes = await listThemes();
    const theme = themes.find(t => t.id === themeId) || themes[0];
    return { themeId: theme.id, theme };
  }
  // existing Harmony Link management API call
}
```

**`setCurrentTheme()` changes:**
```js
export async function setCurrentTheme(themeId) {
  if (isHarmonySpeechEngineMode()) {
    localStorage.setItem('hse-theme', themeId);
    return;
  }
  // existing Harmony Link management API call
}
```

**`listThemes()` changes:**
```js
export async function listThemes() {
  if (isHarmonySpeechEngineMode()) {
    // Return bundled static themes list (same themes available in Harmony Link)
    // OR attempt to load from a bundled themes.json asset
    return BUNDLED_THEMES;  // imported constant
  }
  // existing Harmony Link management API call
}
```

**For the bundled themes:** Either inline the theme definitions as a constant in `themeService.js`, or create a `frontend/src/assets/themes.json` file containing the theme definitions that can be `import`ed. The simplest approach is to copy the themes currently served by the Harmony Link management API into a static JSON asset bundled with the frontend.

---

## Phase 7: Update `HarmonySpeechEngineApp.jsx`

### Files Modified
- [`frontend/src/HarmonySpeechEngineApp.jsx`](frontend/src/HarmonySpeechEngineApp.jsx)

### Changes

**Update TABS enum:**
```jsx
const TABS = {
  TTS: 'tts',
  STT: 'stt',
  VAD: 'vad',
  VC: 'vc',
  AUDIO_RESTORATION: 'audio_restoration',
  SETTINGS: 'settings'
};
```

**Add imports:**
```jsx
import VCHarmonySpeechSettingsView from "./components/modules/VCHarmonySpeechSettingsView.jsx";
import AudioRestorationHarmonySpeechSettingsView from "./components/modules/AudioRestorationHarmonySpeechSettingsView.jsx";
import SpeechEngineSettingsView from "./components/SpeechEngineSettingsView.jsx";
```

**Add initial state for new tabs:**
```jsx
const [vcSettings, setVcSettings] = useState({
  endpoint: "http://localhost:12080",
  model: ""
});

const [audioRestorationSettings, setAudioRestorationSettings] = useState({
  endpoint: "http://localhost:12080",
  model: ""
});
```

**Update navigation tabs:**
- Add "Voice Conversion" tab link (href="#vc")
- Add "Audio Restoration" tab link (href="#audio_restoration")
- Add "Settings" tab link (href="#settings", positioned last / rightmost)
- Add active tab styling: the currently active tab should have a distinct bottom border or background to indicate selection (currently all tabs look the same)

**Update conditional rendering:**
```jsx
{activeTab === TABS.VC && (
  <VCHarmonySpeechSettingsView
    initialSettings={vcSettings}
    saveSettingsFunc={setVcSettings}
  />
)}
{activeTab === TABS.AUDIO_RESTORATION && (
  <AudioRestorationHarmonySpeechSettingsView
    initialSettings={audioRestorationSettings}
    saveSettingsFunc={setAudioRestorationSettings}
  />
)}
{activeTab === TABS.SETTINGS && (
  <SpeechEngineSettingsView />
)}
```

**Add active tab visual indicator:**
Change tab `<a>` className to conditionally apply an active style:
```jsx
className={`inline-block py-2 px-4 font-semibold ${
  activeTab === TABS.TTS
    ? 'text-orange-300 border-b-2 border-orange-400'
    : 'text-orange-400 hover:text-orange-300'
}`}
```

---

## Phase 8: Update `MODULE_DEFAULTS`

### Files Modified
- [`frontend/src/constants/moduleDefaults.js`](frontend/src/constants/moduleDefaults.js)

### Changes

Add new default configuration blocks for the two new module types:

```js
voiceconversion: {
  [PROVIDERS.HARMONYSPEECH]: {
    endpoint: "https://speech.project-harmony.ai",
    model: ""
  }
},
audiorestoration: {
  [PROVIDERS.HARMONYSPEECH]: {
    endpoint: "https://speech.project-harmony.ai",
    model: ""
  }
},
```

---

## File Change Summary

| File | Change Type | Phase |
|------|-------------|-------|
| [`docs/api/openapi.json`](docs/api/openapi.json) | Modify | 0.1 |
| `@harmony-ai/harmonyspeech` npm package | Regenerate + Publish | 0.2–0.3 |
| [`frontend/package.json`](frontend/package.json) | Modify (version bump) | 0.4 |
| [`frontend/src/components/modules/STTHarmonySpeechSettingsView.jsx`](frontend/src/components/modules/STTHarmonySpeechSettingsView.jsx) | Modify | 1 |
| [`frontend/src/components/modules/VADHarmonySpeechSettingsView.jsx`](frontend/src/components/modules/VADHarmonySpeechSettingsView.jsx) | Modify | 2 |
| [`frontend/src/components/modules/VCHarmonySpeechSettingsView.jsx`](frontend/src/components/modules/VCHarmonySpeechSettingsView.jsx) | **New** | 3 |
| [`frontend/src/components/modules/AudioRestorationHarmonySpeechSettingsView.jsx`](frontend/src/components/modules/AudioRestorationHarmonySpeechSettingsView.jsx) | **New** | 4 |
| [`frontend/src/components/SpeechEngineSettingsView.jsx`](frontend/src/components/SpeechEngineSettingsView.jsx) | **New** | 5 |
| [`frontend/src/services/management/themeService.js`](frontend/src/services/management/themeService.js) | Modify | 6 |
| `frontend/src/assets/themes.json` (or inline constant) | **New** | 6 |
| [`frontend/src/HarmonySpeechEngineApp.jsx`](frontend/src/HarmonySpeechEngineApp.jsx) | Modify | 7 |
| [`frontend/src/constants/moduleDefaults.js`](frontend/src/constants/moduleDefaults.js) | Modify | 8 |

---

## Key Technical Notes

### How `DetectVoiceActivityResponse` looks
Based on the protocol, the VAD response contains speech segments. Display as a list of `{start, end, confidence}` objects or as a raw JSON `<pre>` block.

### Voice Conversion source audio handling
Source audio must be read as base64 using `FileReader.readAsDataURL()`, then the `base64Content = dataUrl.split(',')[1]` pattern (same as embedding generation in TTS tab).

### Audio Restoration tab — dependency on Phase 0
The Audio Restoration tab requires the updated npm package with `convertAudio()` and `showAvailableAudioConversionModels()` methods. If Phase 0 is not yet complete, the Audio Restoration tab component can be written to call the API directly via `fetch(${endpoint}/v1/audio/convert, {...})` as a temporary workaround, then updated once the package is published.

### Theme bundling for Speech Engine mode
The Harmony Link management API serves themes from a `themes/` directory. For Speech Engine mode, the simplest approach is to hardcode the same themes as a static `themes.json` asset. Review what themes the Harmony Link management API currently serves and copy those definitions into the static asset.

### CI/CD Note
The UI is deployed from the `harmony-link-ui` repository (per [`docker-release-ui.yml`](.github/workflows/docker-release-ui.yml)). All frontend changes must be made there, not in this repository's `frontend/` directory which is a development checkout. The speech engine repo may need a corresponding CI update once the npm package is published.
