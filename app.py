from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
import requests
import time
import threading
from functools import wraps
import logging
import shutil
from pathlib import Path
import torch.multiprocessing

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up multiprocessing strategy
torch.multiprocessing.set_sharing_strategy('file_system')

# Prevent image bomb errors
Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
CORS(app)  # Allow frontend access

def timeout(seconds):
    """Decorator that adds a timeout to a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                return jsonify({
                    'error': f'Operation timed out after {seconds} seconds. Try with a smaller image or on a more powerful server.',
                    'status': 'timeout'
                }), 503
            
            if error[0] is not None:
                raise error[0]
            
            return result[0]
        return wrapper
    return decorator

def clear_torch_hub_cache():
    """Clear torch hub cache to avoid loading issues"""
    torch_hub_dir = Path.home() / ".cache" / "torch" / "hub"
    if torch_hub_dir.exists():
        # Backup the ultralytics directory first if it exists
        ultralytics_dir = torch_hub_dir / "ultralytics_yolov5_master"
        if ultralytics_dir.exists():
            try:
                shutil.rmtree(ultralytics_dir)
                logger.info(f"✅ Removed {ultralytics_dir}")
            except Exception as e:
                logger.error(f"❌ Failed to remove {ultralytics_dir}: {e}")
    logger.info("🧹 Torch hub cache cleared")

# Model path
MODEL_PATH = 'best.pt'

# Check if model exists
if not os.path.exists(MODEL_PATH):
    logger.error(f"❌ Model file not found: {MODEL_PATH}")
    if os.path.exists('project/yolov5/runs/train/exp3/weights/best.pt'):
        MODEL_PATH = 'project/yolov5/runs/train/exp3/weights/best.pt'
        logger.info(f"✅ Using alternative model path: {MODEL_PATH}")

# Print directory contents for debugging
logger.info(f"📁 Current directory contents: {os.listdir('.')}")

# Load model
logger.info(f"🔄 Loading model from {MODEL_PATH} ...")

# Clear cache before loading
clear_torch_hub_cache()

try:
    # Try loading the model with optimization settings
    model = torch.hub.load('ultralytics/yolov5', 
                         'custom', 
                         path=MODEL_PATH, 
                         force_reload=True, 
                         trust_repo=True,
                         verbose=False)  # Less verbose output
    
    # Optimize for inference
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    logger.info(f"✅ Model loaded with classes: {model.names}")
    
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    # Try alternative loading method
    try:
        import sys
        from pathlib import Path
        
        # Clone YOLOv5 repository if not exists
        yolov5_dir = Path("yolov5_repo")
        if not yolov5_dir.exists():
            logger.info("📥 Cloning YOLOv5 repository...")
            os.system(f"git clone https://github.com/ultralytics/yolov5 {yolov5_dir}")
        
        # Add YOLOv5 directory to path
        sys.path.append(str(yolov5_dir))
        
        # Load model using direct import
        from yolov5_repo.models.experimental import attempt_load
        from yolov5_repo.models.common import AutoShape
        
        device = torch.device('cpu')
        model = attempt_load(MODEL_PATH, device=device)
        model = AutoShape(model)
        logger.info(f"✅ Model loaded with alternative method")
    except Exception as e2:
        logger.error(f"❌ Both model loading methods failed: {str(e2)}")
        raise

# Set up image transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET'])
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plant Disease Detector</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .form-container { margin-top: 20px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 4px; cursor: pointer; }
            #result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; display: none; }
            .loader { border: 5px solid #f3f3f3; border-top: 5px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 2s linear infinite; display: none; margin: 20px auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .image-preview { max-width: 300px; max-height: 300px; margin-top: 15px; border: 1px solid #ddd; display: none; }
        </style>
    </head>
    <body>
        <h1>Plant Disease Detector</h1>
        <p>Upload a plant leaf image to detect diseases</p>
        
        <div class="form-container">
            <form id="upload-form">
                <input type="file" id="image-input" accept="image/*" required>
                <button type="submit">Analyze</button>
            </form>
            <img id="preview" class="image-preview" />
            <div class="loader" id="loader"></div>
        </div>
        
        <div id="result"></div>
        
        <script>
            // Show image preview
            document.getElementById('image-input').addEventListener('change', function(e) {
                const preview = document.getElementById('preview');
                const file = e.target.files[0];
                
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(file);
                }
            });
            
            document.getElementById('upload-form').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const file = document.getElementById('image-input').files[0];
                if (!file) {
                    alert('Please select an image');
                    return;
                }
                
                const loader = document.getElementById('loader');
                const result = document.getElementById('result');
                
                loader.style.display = 'block';
                result.style.display = 'none';
                
                const formData = new FormData();
                formData.append('image', file);
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        result.innerHTML = `
                            <h3>Analysis Results:</h3>
                            <p><strong>Disease:</strong> ${data.disease}</p>
                            <p><strong>Confidence:</strong> ${data.confidence}%</p>
                            <p><strong>Recommended Pesticide:</strong> ${data.pesticide}</p>
                            <p><strong>Recommended Fertilizer:</strong> ${data.fertilizer}</p>
                            <p><em>Processing time: ${data.processing_time || 'N/A'}</em></p>
                        `;
                    } else {
                        result.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                    }
                } catch (error) {
                    result.innerHTML = `<p style="color: red;">Server error: ${error.message}</p>`;
                } finally {
                    loader.style.display = 'none';
                    result.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return html

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'model_loaded': model is not None,
        'model_classes': model.names
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'classes': model.names
    })

@app.route('/predict', methods=['POST'])
@timeout(25)  # Set timeout to 25 seconds
def predict():
    try:
        start_time = time.time()
        logger.info("🔍 Starting prediction request...")
        
        # Get the image from the request
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
            logger.info(f"📸 Image received: {len(img_bytes)} bytes")
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        elif request.is_json and 'image' in request.json:
            img_data = request.json['image']
            # Handle both base64 with and without data URL prefix
            if ',' in img_data:
                img_data = img_data.split(',', 1)[1]
            img_bytes = base64.b64decode(img_data)
            logger.info(f"📸 Base64 image received: {len(img_bytes)} bytes")
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400

        # Resize image to reduce memory usage
        img = img.resize((640, 640), Image.Resampling.LANCZOS)
        
        # Run inference with optimizations
        logger.info("🧠 Running inference...")
        with torch.no_grad():  # Disable gradient calculations
            results = model(img, size=640)
        
        # Process results
        predictions = results.pandas().xyxy[0]
        logger.info(f"📊 Got predictions: {len(predictions)} items")
        
        if len(predictions) > 0:
            best_pred = predictions.sort_values('confidence', ascending=False).iloc[0]
            disease = best_pred['name']
            confidence = round(float(best_pred['confidence']) * 100, 2)
        else:
            disease = "Healthy"
            confidence = 100.0
        
        elapsed_time = time.time() - start_time
        logger.info(f"🧪 Prediction: {disease} ({confidence}%) took {elapsed_time:.2f}s")

        # Disease-specific recommendations
        pesticide_recommendations = {
            "Bacterial_spot": "Use copper-based bactericides like copper hydroxide or copper sulfate. Apply every 7-10 days during wet weather.",
            "Black_rot": "Apply myclobutanil, thiophanate-methyl, or captan fungicides. Start at bud break and continue at 10-14 day intervals.",
            "Late_Blight": "Use chlorothalonil, mancozeb, or copper-based fungicides. Apply preventatively before symptoms appear.",
            "Powder_Mildew": "Apply sulfur, potassium bicarbonate, or neem oil. Treat at first sign of disease and repeat every 7-10 days.",
            "Healthy": "No pesticide needed. Continue regular monitoring."
        }

        fertilizer_recommendations = {
            "Bacterial_spot": "Low nitrogen, high phosphorus and potassium (e.g., 5-10-10). Add calcium to strengthen cell walls.",
            "Black_rot": "Balanced NPK fertilizer (e.g., 10-10-10) with added calcium. Avoid excessive nitrogen.",
            "Late_Blight": "Low nitrogen, high potassium fertilizer (e.g., 5-10-15). Potassium helps build disease resistance.",
            "Powder_Mildew": "Balanced fertilizer with silicon supplements. Avoid excessive nitrogen which promotes susceptible new growth.",
            "Healthy": "Standard balanced fertilizer appropriate for plant type. Follow recommended application rates."
        }

        pesticide = pesticide_recommendations.get(disease, "General purpose fungicide appropriate for the plant type")
        fertilizer = fertilizer_recommendations.get(disease, "Balanced NPK fertilizer appropriate for the plant type")

        return jsonify({
            'disease': disease,
            'confidence': confidence,
            'pesticide': pesticide,
            'fertilizer': fertilizer,
            'processing_time': f"{elapsed_time:.2f}s"
        })

    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Add an alias for the prediction endpoint for compatibility
@app.route('/predict/predict', methods=['POST'])
def predict_alias():
    return predict()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port)
