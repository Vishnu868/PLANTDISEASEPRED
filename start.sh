#!/bin/bash

echo "📥 Downloading model from Google Drive..."
python download_model.py

if [ ! -d "yolov5" ]; then
  echo "📥 Cloning YOLOv5 repository..."
  git clone https://github.com/ultralytics/yolov5
else
  echo "📁 YOLOv5 already exists, skipping clone."
fi

echo "📦 Installing dependencies..."
pip install -r yolov5/requirements.txt
pip install -r requirements.txt

echo "🚀 Starting Flask server..."
python app.py
