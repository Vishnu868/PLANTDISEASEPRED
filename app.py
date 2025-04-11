from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import os
import numpy as np

# Patch for YOLOv5 models trained on Windows, now running on Linux (like Render)
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath
# Prevent image bomb errors
Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
CORS(app)  # Allow frontend (e.g. Flutter, React) to access this server

# Load your trained model - use a locally cloned YOLOv5 repository
MODEL_PATH = 'best.pt'  # Will look for the model in the current directory
print(f"🔄 Loading model from {MODEL_PATH} ...")

try:
    # Import YOLOv5 modules from local repo
    import sys
    sys.path.append('./yolov5')  # Add the cloned repo to path
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    
    # Initialize device and load model
    device = select_device('')  # Use CPU by default
    model = DetectMultiBackend(MODEL_PATH, device=device)
    model.eval()
    print(f"✅ Model loaded with classes: {model.names}")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
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

# ✅ ORIGINAL ENDPOINT
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'model_loaded': model is not None
    })

# ✅ DUPLICATED ENDPOINT FOR FLUTTER COMPATIBILITY
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

# ✅ ORIGINAL PREDICT ENDPOINT
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
        # Or check if it's base64
        elif request.is_json and 'image' in request.json:
            img_data = request.json['image']
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400

        # Prediction using the local model
        img_size = check_img_size(640, s=model.stride)  # Determine letterbox size
        img_array = np.array(img)
        img_processed = letterbox(img_array, img_size, stride=model.stride, auto=True)[0]
        img_processed = img_processed.transpose((2, 0, 1))[::-1]  # BGR to RGB, to 3xHxW
        img_processed = np.ascontiguousarray(img_processed)
        img_tensor = torch.from_numpy(img_processed).float().to(device) / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Run inference
        pred = model(img_tensor)
        
        # Process results
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
        results = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img_array.shape).round()
                for *xyxy, conf, cls in det:
                    results.append({
                        'name': model.names[int(cls)],
                        'confidence': float(conf),
                        'xyxy': [float(x) for x in xyxy]
                    })

        # If predictions are found
        if results:
            best_pred = max(results, key=lambda x: x['confidence'])
            disease = best_pred['name']
            confidence = round(best_pred['confidence'] * 100, 2)
        else:
            disease = "Healthy"
            confidence = 100.0

        # Debug log (optional)
        print(f"🧪 Prediction: {disease} ({confidence}%)")
        
        # Treatment recommendations (customize these)
        pesticide_recommendations = {
            "Apple Scab": "Use captan or myclobutanil fungicides",
            "Black Spot": "Apply neem oil or chlorothalonil",
            "Tomato Late Blight": "Use copper-based fungicides",
            "Powdery Mildew": "Apply sulfur-based fungicides",
            "Healthy": "No pesticide needed"
        }
        
        fertilizer_recommendations = {
            "Apple Scab": "Use balanced NPK with calcium",
            "Black Spot": "Rose-specific fertilizer with magnesium",
            "Tomato Late Blight": "Low nitrogen, high potassium fertilizer",
            "Powdery Mildew": "Balanced fertilizer with silica",
            "Healthy": "Standard balanced fertilizer"
        }
        
        # Get recommendations or use fallback
        pesticide = pesticide_recommendations.get(disease, "General purpose fungicide")
        fertilizer = fertilizer_recommendations.get(disease, "Balanced NPK fertilizer")

        return jsonify({
            'disease': disease,
            'confidence': confidence,
            'pesticide': pesticide,
            'fertilizer': fertilizer
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ✅ DUPLICATED ENDPOINT FOR FLUTTER COMPATIBILITY
@app.route('/predict/predict', methods=['POST'])
def predict_alias():
    return predict()      

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
