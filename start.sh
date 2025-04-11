#!/bin/bash

echo "📥 Downloading model from Google Drive..."
python download_model.py

# Check if model was downloaded successfully
if [ ! -f "best.pt" ] || [ ! -s "best.pt" ]; then
    echo "❌ Model download failed or file is empty. Please check the download URL."
    exit 1
fi

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Starting Flask server..."
# Use gunicorn for production deployment
gunicorn app:app --bind 0.0.0.0:$PORT
