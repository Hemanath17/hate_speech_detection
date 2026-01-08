#!/bin/bash
# Start the hate speech detection API server

cd "$(dirname "$0")/../.."

# Activate virtual environment
source hate_env/bin/activate

# Install dependencies if needed
pip install -q -r src/api/requirements.txt

# Start the server
echo "Starting Hate Speech Detection API..."
echo "Server will run on http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python src/api/app.py

