#!/bin/bash
echo "📥 Downloading model from Google Drive..."
python download_model.py

echo "🚀 Starting Flask server..."
python app.py
