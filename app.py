from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import io
import base64
import os
import sys
import pathlib
import logging
import time
from PIL import Image
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prevent image bomb errors
Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
CORS(app)  # Allow frontend (e.g. Flutter, React) to access this server

# Load your trained model
# Inside app.py, replace the model loading part with:

MODEL_PATH = 'trained_model.h5'
FALLBACK_PATH = 'best.pt'
logger.info(f"🔄 Loading model from {MODEL_PATH} or fallback {FALLBACK_PATH}...")

try:
    # Custom loading approach for compatibility
    def load_model_with_custom_objects():
        model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_PATH
        
        # Try different loading strategies
        try:
            # Strategy 1: Standard load with compile=False
            logger.info("Trying standard loading with compile=False...")
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e1:
            logger.warning(f"Standard loading failed: {str(e1)}")
            
            try:
                # Strategy 2: Load with custom objects
                logger.info("Trying loading with custom objects...")
                return tf.keras.models.load_model(
                    model_path, 
                    custom_objects={'InputLayer': tf.keras.layers.InputLayer},
                    compile=False
                )
            except Exception as e2:
                logger.warning(f"Custom objects loading failed: {str(e2)}")
                
                try:
                    # Strategy 3: Try loading the model architecture and weights separately
                    logger.info("Trying to load model structure from JSON...")
                    json_path = model_path.replace('.h5', '.json').replace('.keras', '.json')
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            model_json = f.read()
                        model = tf.keras.models.model_from_json(model_json)
                        model.load_weights(model_path)
                        return model
                    else:
                        raise FileNotFoundError(f"No JSON architecture file found at {json_path}")
                except Exception as e3:
                    # Last resort - try loading with legacy mode
                    logger.warning(f"JSON loading failed: {str(e3)}")
                    logger.info("Trying legacy H5 loading as last resort...")
                    return tf.keras.models.load_model(model_path, compile=False, options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'))
    
    # Try loading with our custom approach
    model = load_model_with_custom_objects()
    logger.info(f"✅ Model loaded successfully with {len(class_names)} classes")
    
except Exception as e:
    logger.error(f"❌ All model loading attempts failed: {str(e)}")
    model = None

# Define class names for the model
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

try:
    # Try multiple approaches to load the model
    logger.info(f"Attempting to load model from {MODEL_PATH}...")
    
    # First try: Standard Keras model loading regardless of file extension
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"✅ Model loaded with {len(class_names)} classes via standard loading")
    except Exception as e1:
        logger.warning(f"Standard model loading failed: {str(e1)}")
        
        # Second try: If the file is actually a Keras model with .pt extension
        try:
            # Check if actual_model_path exists or use MODEL_PATH
            actual_model_path = 'trained_model.keras' if os.path.exists('trained_model.keras') else MODEL_PATH
            logger.info(f"Attempting to load from alternate path: {actual_model_path}")
            model = tf.keras.models.load_model(actual_model_path)
            logger.info(f"✅ Model loaded with {len(class_names)} classes via alternate path")
        except Exception as e2:
            logger.error(f"Alternate model loading failed: {str(e2)}")
            
            # Third try: Custom objects and compile=False
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                logger.info(f"✅ Model loaded with compile=False")
            except Exception as e3:
                logger.error(f"All model loading attempts failed")
                model = None
                raise Exception(f"Could not load model: {str(e1)}, {str(e2)}, {str(e3)}")
            
except Exception as e:
    logger.error(f"❌ Error loading model: {str(e)}")
    model = None

# Define image preprocessing for Keras model
def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for Keras model prediction"""
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET'])
def index():
    model_status = "✅ Loaded" if model is not None else "❌ Not loaded"
    return jsonify({
        'status': '🟢 Server Running',
        'model_status': model_status,
        'predict_endpoint': '/predict',
        'classes_endpoint': '/classes',
        'model_path': MODEL_PATH,
        'classes': class_names
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
        'classes': class_names
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
        
        # Preprocess image for Keras model
        img_processed = preprocess_image(img)
        
        # Run inference
        logger.info(f"Running inference on image")
        predictions = model.predict(img_processed)
        
        # Get the class with highest probability
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        disease = class_names[predicted_class_idx]
        
        logger.info(f"Prediction: {disease} with confidence {confidence:.2f}%")
        
        # Check if the prediction is a "healthy" class
        is_healthy = "healthy" in disease.lower()
        
        # Treatment recommendations (customize these)
        pesticide_recommendations = {
            "Apple___Apple_scab": "Use captan or myclobutanil fungicides",
            "Apple___Black_rot": "Apply copper-based fungicides or thiophanate-methyl",
            "Apple___Cedar_apple_rust": "Use propiconazole or myclobutanil",
            "Tomato___Late_blight": "Use copper-based fungicides or chlorothalonil",
            "Tomato___Early_blight": "Apply mancozeb or chlorothalonil",
            "Tomato___Bacterial_spot": "Use copper-based bactericides",
            "Tomato___Leaf_Mold": "Apply chlorothalonil or copper-based fungicides",
            "Tomato___Septoria_leaf_spot": "Use chlorothalonil or copper-based fungicides",
            "Tomato___Spider_mites Two-spotted_spider_mite": "Apply insecticidal soap or neem oil",
            "Tomato___Target_Spot": "Use chlorothalonil or azoxystrobin",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies with insecticidal soap",
            "Tomato___Tomato_mosaic_virus": "No chemical treatment; remove infected plants",
            "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur-based fungicides or neem oil",
            "Grape___Black_rot": "Apply mancozeb or myclobutanil",
            "Grape___Esca_(Black_Measles)": "No effective chemical control; remove infected vines",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply mancozeb or copper-based fungicides",
            "Orange___Haunglongbing_(Citrus_greening)": "No effective chemical treatment; control psyllids",
            "Peach___Bacterial_spot": "Apply copper-based bactericides early in season",
            "Pepper,_bell___Bacterial_spot": "Use copper-based bactericides with mancozeb",
            "Potato___Early_blight": "Apply chlorothalonil or azoxystrobin",
            "Potato___Late_blight": "Use mancozeb or chlorothalonil",
            "Squash___Powdery_mildew": "Apply sulfur-based fungicides or neem oil",
            "Strawberry___Leaf_scorch": "Use captan or thiophanate-methyl",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Apply azoxystrobin or pyraclostrobin",
            "Corn_(maize)___Common_rust_": "Use azoxystrobin or pyraclostrobin",
            "Corn_(maize)___Northern_Leaf_Blight": "Apply propiconazole or azoxystrobin"
        }
        
        # Add "healthy" recommendations for all plants
        for class_name in class_names:
            if "healthy" in class_name.lower():
                pesticide_recommendations[class_name] = "No pesticide needed"
        
        fertilizer_recommendations = {
            "Apple___Apple_scab": "Use balanced NPK with calcium",
            "Apple___Black_rot": "Use calcium-rich fertilizer with balanced NPK",
            "Apple___Cedar_apple_rust": "Balanced NPK with micronutrients",
            "Tomato___Late_blight": "Low nitrogen, high potassium fertilizer",
            "Tomato___Early_blight": "Balanced NPK with calcium",
            "Tomato___Bacterial_spot": "Calcium-rich fertilizer with micronutrients",
            "Tomato___Leaf_Mold": "Well-balanced fertilizer with proper drainage",
            "Tomato___Septoria_leaf_spot": "Balanced fertilizer with calcium",
            "Tomato___Spider_mites Two-spotted_spider_mite": "Standard balanced fertilizer, avoid over-fertilizing",
            "Tomato___Target_Spot": "Balanced NPK with calcium",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Balanced fertilizer with micronutrients",
            "Tomato___Tomato_mosaic_virus": "Well-balanced fertilizer to support plant health",
            "Cherry_(including_sour)___Powdery_mildew": "Balanced fertilizer with silica",
            "Grape___Black_rot": "Balanced NPK with calcium",
            "Grape___Esca_(Black_Measles)": "Balanced fertilizer with micronutrients",
            "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Balanced NPK with magnesium",
            "Orange___Haunglongbing_(Citrus_greening)": "High quality citrus fertilizer with micronutrients",
            "Peach___Bacterial_spot": "Balanced NPK with calcium",
            "Pepper,_bell___Bacterial_spot": "Balanced fertilizer with calcium",
            "Potato___Early_blight": "Low nitrogen, balanced potassium and phosphorus",
            "Potato___Late_blight": "Low nitrogen, high potassium fertilizer",
            "Squash___Powdery_mildew": "Balanced fertilizer with silica",
            "Strawberry___Leaf_scorch": "Balanced fertilizer with calcium",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Balanced NPK with micronutrients",
            "Corn_(maize)___Common_rust_": "Balanced NPK with zinc",
            "Corn_(maize)___Northern_Leaf_Blight": "Balanced NPK with proper micronutrients"
        }
        
        # Add "healthy" recommendations for all plants
        for class_name in class_names:
            if "healthy" in class_name.lower():
                fertilizer_recommendations[class_name] = "Standard balanced fertilizer"
        
        # Get recommendations or use fallback
        pesticide = pesticide_recommendations.get(disease, "General purpose fungicide")
        fertilizer = fertilizer_recommendations.get(disease, "Balanced NPK fertilizer")

        response_data = {
            'disease': disease,
            'is_healthy': is_healthy,
            'confidence': round(confidence, 2),
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
    # Make sure to use the PORT environment variable provided by Render
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Flask server on port {port}...")
    # THIS IS CRITICAL: Make sure to bind to 0.0.0.0 for Render to detect the port
    app.run(host='0.0.0.0', port=port)
