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

# Patch for YOLOv5 models trained on Windows, now running on Linux (like Render)
if sys.platform != "win32":
    pathlib.WindowsPath = pathlib.PosixPath

# Prevent image bomb errors
Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend apps (Flutter, React, etc.)

# Load YOLOv5 model
MODEL_PATH = 'best.pt'
print(f"🔄 Loading model from {MODEL_PATH} ...")
print("📁 Current directory contents:", os.listdir('.'))

try:
    sys.path.append('./yolov5')
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.augmentations import letterbox

    device = select_device('')  # CPU
    model = DetectMultiBackend(MODEL_PATH, device=device)
    model.eval()
    print(f"✅ Model loaded with classes: {model.names}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': '🟢 Server Running',
        'model_status': "✅ Loaded" if model else "❌ Not loaded",
        'predict_endpoint': '/predict',
        'classes_endpoint': '/classes',
        'model_path': MODEL_PATH,
        'classes': model.names if model else []
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
        return jsonify({'error': 'Model not loaded', 'classes': []}), 503
    return jsonify({'classes': model.names})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    try:
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        elif request.is_json and 'image' in request.json:
            img_data = request.json['image']
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400

        img.save("debug_uploaded.jpg")
        print("🖼️ Uploaded image shape:", np.array(img).shape)

        img_size = check_img_size(640, s=model.stride)
        img_array = np.array(img)
        img_processed = letterbox(img_array, img_size, stride=model.stride, auto=True)[0]
        print("📐 Letterboxed image shape:", img_processed.shape)
        Image.fromarray(img_processed).save("letterboxed.jpg")

        img_processed = img_processed.transpose((2, 0, 1))[::-1]  # BGR -> RGB
        img_processed = np.ascontiguousarray(img_processed)
        img_tensor = torch.from_numpy(img_processed).float().to(device) / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        pred = model(img_tensor)
        pred = non_max_suppression(pred, conf_thres=0.15, iou_thres=0.45)

        print("🎯 Prediction tensor:", pred)

        results = []
        for i, det in enumerate(pred):
            if det is not None and len(det):
                print(f"🧠 Detected {len(det)} objects in image {i}")
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img_array.shape).round()
                for *xyxy, conf, cls in det:
                    cls_name = model.names[int(cls)]
                    print(f"🔍 Class: {cls_name}, Conf: {conf.item():.4f}")
                    results.append({
                        'name': cls_name,
                        'confidence': float(conf),
                        'xyxy': [float(x) for x in xyxy]
                    })
            else:
                print(f"🛑 No detections in image {i}")

        if results:
            best_pred = max(results, key=lambda x: x['confidence'])
            disease = best_pred['name']
            confidence = round(best_pred['confidence'] * 100, 2)
        else:
            disease = "Healthy"
            confidence = 100.0

        print(f"🧪 Final Prediction: {disease} ({confidence}%)")

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

        pesticide = pesticide_recommendations.get(disease, "General purpose fungicide")
        fertilizer = fertilizer_recommendations.get(disease, "Balanced NPK fertilizer")

        return jsonify({
            'disease': disease,
            'confidence': confidence,
            'pesticide': pesticide,
            'fertilizer': fertilizer,
            'raw_detections': str(pred)  # Debug only
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/predict', methods=['POST'])
def predict_alias():
    return predict()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
