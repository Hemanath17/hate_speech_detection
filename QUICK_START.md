# Quick Start Guide - Fix API Server Error

## The Error
```
[Hate Speech Detector] API server is not running.
[Hate Speech Detector] Start server: python src/api/app.py
```

## Solution: Start the API Server

### Option 1: Using the Start Script (Easiest)
```bash
cd /Users/hemanatharumugam/Documents/Projects/Hatespeech
./start_api.sh
```

### Option 2: Manual Start
```bash
cd /Users/hemanatharumugam/Documents/Projects/Hatespeech
source hate_env/bin/activate
python src/api/app.py
```

### Option 3: Run in Background (Recommended)
```bash
cd /Users/hemanatharumugam/Documents/Projects/Hatespeech
source hate_env/bin/activate
nohup python src/api/app.py > api.log 2>&1 &
```

## Verify API is Running

After starting, check if it's working:
```bash
curl http://localhost:5000/health
```

Should return:
```json
{"device":"cpu","model_loaded":true,"status":"healthy"}
```

## Then Test Extension

1. **Reload Extension**: Go to `chrome://extensions/` and reload
2. **Go to Reddit**: Visit any Reddit post with comments
3. **Check Console**: Should see `✓ API server connected!` instead of error

## Troubleshooting

### Port 5000 Already in Use
```bash
# Find what's using port 5000
lsof -ti:5000

# Kill it if needed
kill -9 $(lsof -ti:5000)

# Then start API again
```

### Model Not Loading
- Check that `models/mbert_improved_chinese_multilingual/` exists
- Check file permissions
- Verify model files are present

### Still Not Working?
1. Check API logs for errors
2. Verify virtual environment is activated
3. Make sure all dependencies are installed: `pip install -r requirements.txt`

