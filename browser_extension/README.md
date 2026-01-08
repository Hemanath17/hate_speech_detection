# Hate Speech Detector Browser Extension

Real-time hate speech detection browser extension supporting 5 languages (English, Tamil, Hindi, Chinese, Spanish).

## Features

- ✅ **Real-time Detection**: Detects hate speech as you type
- ✅ **Visual Highlighting**: Highlights detected hate speech in red/yellow
- ✅ **Pre-submission Flagging**: Warns before posting hate speech
- ✅ **Multi-language Support**: English, Tamil, Hindi, Chinese, Spanish
- ✅ **Smart Detection**: Uses optimized thresholds per language

## Installation

### 1. Start the Backend API

```bash
cd /Users/hemanatharumugam/Documents/Projects/Hatespeech
source hate_env/bin/activate
pip install -r src/api/requirements.txt
python src/api/app.py
```

The API will run on `http://localhost:5000`

### 2. Load Extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `browser_extension` folder
5. Extension is now installed!

### 3. Load Extension in Firefox

1. Open Firefox and go to `about:debugging`
2. Click "This Firefox"
3. Click "Load Temporary Add-on"
4. Select `browser_extension/manifest.json`
5. Extension is now installed!

## Usage

1. **Enable/Disable**: Click the extension icon to toggle detection
2. **Real-time Detection**: Type in any input field - hate speech will be highlighted
3. **Pre-submission Warning**: When submitting forms with hate speech, you'll see a warning
4. **Test Detection**: Use the popup to test detection on custom text

## Configuration

- **API URL**: Default is `http://localhost:5000/api/detect`
- **Debounce Delay**: 500ms (waits after typing stops)
- **Thresholds**: Language-specific (loaded from `models/optimized_thresholds.json`)

## File Structure

```
browser_extension/
├── manifest.json          # Extension manifest
├── background/
│   └── background.js      # Service worker
├── content/
│   ├── content.js         # Content script (main logic)
│   └── content.css        # Highlighting styles
├── popup/
│   ├── popup.html         # Popup UI
│   ├── popup.js           # Popup logic
│   └── popup.css          # Popup styles
└── assets/
    └── icons/             # Extension icons
```

## API Endpoints

- `GET /health` - Health check
- `POST /api/detect` - Detect hate speech in text
- `POST /api/detect-batch` - Detect hate speech in multiple texts

## Troubleshooting

1. **Extension not working**: Check if API server is running
2. **No highlights**: Check browser console for errors
3. **API errors**: Verify API is accessible at `http://localhost:5000`

## Development

- Content script runs on all pages
- Uses MutationObserver to detect new input fields
- Debounces API calls to avoid rate limiting
- Caches detection results for performance

