from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
import numpy as np
import sys
import pathlib
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Patch for YOLOv5 models trained on Windows, now running on Linux (like Render)
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath
# Prevent image bomb errors
Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
CORS(app)  # Allow frontend (e.g. Flutter, React) to access this server

# Load your trained model - use a locally cloned YOLOv5 repository
MODEL_PATH = 'best.pt'  # Will look for the model in the current directory
logger.info(f"🔄 Loading model from {MODEL_PATH} ...")

try:
    # Import YOLOv5 modules from local repo
    sys.path.append('./yolov5')  # Add the cloned repo to path
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    
    # Initialize device and load model
    device = select_device('')  # Use CPU by default
    model = DetectMultiBackend(MODEL_PATH, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size((640, 640), s=stride)  # Check image size
    model.eval()
    logger.info(f"✅ Model loaded with classes: {model.names}")
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    # Set a fallback if model can't be loaded - will be overridden if model loads later
    model = None

# Define image transformation (not needed by YOLO, but in case you use it elsewhere)
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET'])
def index():
    model_status = "✅ Loaded" if model is not None else "❌ Not loaded"
    classes = model.names if model is not None else []
    return jsonify({
        'status': '🟢 Server Running',
        'model_status': model_status,
        'predict_endpoint': '/predict',
        'classes_endpoint': '/classes',
        'model_path': MODEL_PATH,
        'classes': classes
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'model_loaded': model is not None
    })

@app.route('/predict/health', methods=['GET'])
def health_check_alias():
    return health_check()

@app.route('/classes', methods=['GET'])
def get_classes():
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'classes': []
        }), 503
    return jsonify({
        'classes': model.names
    })

@app.route('/debug-image', methods=['POST'])
def debug_image():
    """
    Debug endpoint to save and check received images
    """
    try:
        # Get the image
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
        elif request.is_json and 'image' in request.json:
            img_data = request.json['image']
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))
        else:
            return jsonify({'error': 'No image provided'}), 400
            
        # Save the image
        filename = f"debug_image_{int(time.time())}.png"
        img.save(filename)
        
        # Return image info
        return jsonify({
            'message': f'Image saved as {filename}',
            'size': img.size,
            'format': img.format
        })
    except Exception as e:
        logger.error(f"Error saving debug image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
        
    try:
        # Check if image is sent as a file
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            logger.info(f"Image received as file: {file.filename}")
        # Or check if it's base64
        elif request.is_json and 'image' in request.json:
            img_data = request.json['image']
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            logger.info("Image received as base64")
        else:
            return jsonify({'error': 'No image provided'}), 400

        # Log image details for debugging
        logger.info(f"Original image size: {img.size}, format: {img.format if hasattr(img, 'format') else 'unknown'}")
        
        # Save a copy of the original image for debugging
        debug_filename = f"debug_original_{int(time.time())}.png"
        img.save(debug_filename)
        logger.info(f"Saved debug image to {debug_filename}")
        
        # Prediction using the local model
        img_array = np.array(img)
        
        # Process image for YOLO
        img_processed = letterbox(img_array, imgsz, stride=stride, auto=True)[0]
        img_processed = img_processed.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3xHxW
        img_processed = np.ascontiguousarray(img_processed)
        img_tensor = torch.from_numpy(img_processed).float().to(device) / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Run inference with debug info
        logger.info(f"Running inference on image of shape: {img_array.shape}")
        with torch.no_grad():  # Ensure no gradients are computed
            pred = model(img_tensor)
        
        # Process results with lower confidence threshold for debugging
        pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45)  # Lower threshold to catch more predictions
        
        results = []
        for i, det in enumerate(pred):
            logger.info(f"Detections: {len(det)}")
            if len(det):
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img_array.shape).round()
                for *xyxy, conf, cls in det:
                    class_idx = int(cls)
                    class_name = model.names[class_idx]
                    confidence = float(conf)
                    logger.info(f"Detection: {class_name} with confidence {confidence:.2f}")
                    results.append({
                        'name': class_name,
                        'confidence': confidence,
                        'xyxy': [float(x) for x in xyxy]
                    })

        # If predictions are found
        if results:
            best_pred = max(results, key=lambda x: x['confidence'])
            disease = best_pred['name']
            confidence = round(best_pred['confidence'] * 100, 2)
            logger.info(f"Best prediction: {disease} with confidence {confidence}%")
        else:
            # Only classify as healthy if explicitly detected or no detections at all
            if "Healthy" in model.names:
                disease = "Healthy"
                confidence = 90.0  # Lower confidence for default case
            else:
                disease = "Unknown"
                confidence = 0.0
            logger.info(f"No detections found, defaulting to: {disease}")

        # Treatment recommendations (customize these)
        pesticide_recommendations = {
            "Apple_scab": "Use captan or myclobutanil fungicides",
            "Black_rot": "Apply copper-based fungicides or thiophanate-methyl",
            "Cedar_apple_rust": "Use propiconazole or myclobutanil",
            "Tomato_Late_blight": "Use copper-based fungicides or chlorothalonil",
            "Tomato_Early_blight": "Apply mancozeb or chlorothalonil",
            "Tomato_Bacterial_spot": "Use copper-based bactericides",
            "Tomato_Leaf_Mold": "Apply chlorothalonil or copper-based fungicides",
            "Tomato_Septoria_leaf_spot": "Use chlorothalonil or copper-based fungicides",
            "Tomato_Spider_mites": "Apply insecticidal soap or neem oil",
            "Tomato_Target_Spot": "Use chlorothalonil or azoxystrobin",
            "Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with insecticidal soap",
            "Tomato_mosaic_virus": "No chemical treatment; remove infected plants",
            "Powdery_mildew": "Apply sulfur-based fungicides or neem oil",
            "Healthy": "No pesticide needed",
            # Add other diseases according to your model's classes
        }
        
        fertilizer_recommendations = {
            "Apple_scab": "Use balanced NPK with calcium",
            "Black_rot": "Use calcium-rich fertilizer with balanced NPK",
            "Cedar_apple_rust": "Balanced NPK with micronutrients",
            "Tomato_Late_blight": "Low nitrogen, high potassium fertilizer",
            "Tomato_Early_blight": "Balanced NPK with calcium",
            "Tomato_Bacterial_spot": "Calcium-rich fertilizer with micronutrients",
            "Tomato_Leaf_Mold": "Well-balanced fertilizer with proper drainage",
            "Tomato_Septoria_leaf_spot": "Balanced fertilizer with calcium",
            "Tomato_Spider_mites": "Standard balanced fertilizer, avoid over-fertilizing",
            "Tomato_Target_Spot": "Balanced NPK with calcium",
            "Tomato_Yellow_Leaf_Curl_Virus": "Balanced fertilizer with micronutrients",
            "Tomato_mosaic_virus": "Well-balanced fertilizer to support plant health",
            "Powdery_mildew": "Balanced fertilizer with silica",
            "Healthy": "Standard balanced fertilizer",
            # Add more recommendations based on your model's classes
        }
        
        # Get recommendations or use fallback
        pesticide = pesticide_recommendations.get(disease, "General purpose fungicide")
        fertilizer = fertilizer_recommendations.get(disease, "Balanced NPK fertilizer")

        response_data = {
            'disease': disease,
            'confidence': confidence,
            'pesticide': pesticide,
            'fertilizer': fertilizer,
            'debug_image': debug_filename
        }
        
        logger.info(f"Returning prediction: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Error in prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/predict/predict', methods=['POST'])
def predict_alias():
    return predict()      

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
