#!/bin/bash
echo "🔧 Starting deployment process..."

# Download model
echo "📥 Downloading model from Google Drive..."
python download_model.py

# Check if model was downloaded successfully
if [ ! -f "best.pt" ] || [ ! -s "best.pt" ]; then
    echo "❌ Model download failed or file is empty. Please check the download URL."
    exit 1
fi

echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Set environment variables for TensorFlow
echo "🔧 Configuring TensorFlow..."
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_VISIBLE_DEVICES=-1  # Force CPU mode in case GPU is detected but problematic

echo "🚀 Starting Flask server..."
# Use gunicorn for production deployment
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
