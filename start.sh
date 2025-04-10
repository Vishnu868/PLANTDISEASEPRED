#!/bin/bash
echo "📥 Downloading model from Google Drive..."
python download_model.py

echo "📥 Cloning YOLOv5 repository..."
git clone https://github.com/ultralytics/yolov5

echo "🚀 Starting Flask server..."
python app.py
