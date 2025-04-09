from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

# Prevent image bomb errors
Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
CORS(app)  # Allow frontend (e.g. Flutter, React) to access this server

# Load your trained model
MODEL_PATH = 'D:/IOT/best.pt'  # Update this if needed
print(f"🔄 Loading model from {MODEL_PATH} ...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
model.eval()
print(f"✅ Model loaded with classes: {model.names}")

# Define image transformation (not needed by YOLO, but in case you use it elsewhere)
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': '🟢 Server Running',
        'predict_endpoint': '/predict',
        'classes_endpoint': '/classes',
        'model_path': MODEL_PATH
    })

# ✅ ORIGINAL ENDPOINT
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Server is running'
    })

# ✅ DUPLICATED ENDPOINT FOR FLUTTER COMPATIBILITY
@app.route('/predict/health', methods=['GET'])
def health_check_alias():
    return health_check()

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'classes': model.names
    })

# ✅ ORIGINAL PREDICT ENDPOINT
@app.route('/predict', methods=['POST'])
def predict():
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

        # Prediction
        results = model(img)
        predictions = results.pandas().xyxy[0]  # Convert to DataFrame

        # If predictions are found
        if len(predictions) > 0:
            best_pred = predictions.sort_values('confidence', ascending=False).iloc[0]
            disease = best_pred['name']
            confidence = round(float(best_pred['confidence']) * 100, 2)
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
    
