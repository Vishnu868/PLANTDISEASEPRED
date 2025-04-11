#!/bin/bash

echo "📥 Downloading model from Google Drive..."
python download_model.py

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Starting Flask server..."
# Use the PORT environment variable for compatibility with hosting platforms
PORT=${PORT:-10000}
python app.py
