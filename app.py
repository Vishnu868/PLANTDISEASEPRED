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
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prevent image bomb errors
Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__)
CORS(app)  # Allow frontend (e.g. Flutter, React) to access this server

# Define class names for the model FIRST, before model loading
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

# Consolidated model loading (removing duplicated loading attempts)
MODEL_PATH = 'trained_model.h5'
FALLBACK_PATH = 'best.pt'
ALTERNATE_PATH = 'trained_model.keras'
model = None
target_height = 128  # Default target height
target_width = 128   # Default target width

logger.info(f"🔄 Loading model from {MODEL_PATH} or fallbacks...")

try:
    # First try: Standard Keras model loading
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info(f"✅ Model loaded with {len(class_names)} classes via standard loading")
        except Exception as e1:
            logger.warning(f"Standard model loading failed: {str(e1)}")
            
            # Try with compile=False
            try:
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                logger.info(f"✅ Model loaded with compile=False")
            except Exception as e2:
                logger.warning(f"Loading with compile=False failed: {str(e2)}")
    
    # If model still not loaded, try alternate paths
    if model is None and os.path.exists(ALTERNATE_PATH):
        try:
            logger.info(f"Attempting to load from alternate path: {ALTERNATE_PATH}")
            model = tf.keras.models.load_model(ALTERNATE_PATH)
            logger.info(f"✅ Model loaded from alternate path")
        except Exception as e3:
            logger.warning(f"Alternate path loading failed: {str(e3)}")
    
    # If still not loaded, try fallback
    if model is None and os.path.exists(FALLBACK_PATH):
        try:
            logger.info(f"Attempting to load from fallback path: {FALLBACK_PATH}")
            model = tf.keras.models.load_model(FALLBACK_PATH, compile=False)
            logger.info(f"✅ Model loaded from fallback path")
        except Exception as e4:
            logger.warning(f"Fallback path loading failed: {str(e4)}")
    
    # Last resort - try with custom objects
    if model is None:
        available_paths = [p for p in [MODEL_PATH, ALTERNATE_PATH, FALLBACK_PATH] if os.path.exists(p)]
        if available_paths:
            try:
                model_path = available_paths[0]
                logger.info(f"Trying loading with custom objects from {model_path}...")
                model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects={'InputLayer': tf.keras.layers.InputLayer},
                    compile=False
                )
                logger.info(f"✅ Model loaded with custom objects")
            except Exception as e5:
                logger.warning(f"Custom objects loading failed: {str(e5)}")
                
    if model is None:
        logger.error("❌ All model loading attempts failed")
    else:
        # Check model input shape to dynamically set target size
        try:
            input_shape = model.input_shape
            if input_shape and len(input_shape) == 4:
                target_height = input_shape[1]
                target_width = input_shape[2]
                logger.info(f"✓ Model expects input shape: {input_shape}, setting target size to ({target_height}, {target_width})")
                
            # Set TensorFlow session config if possible
            try:
                # For TF 2.x compatibility
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("Set GPU memory growth")
                
                # Optimize for inference
                model = tf.function(model)
                model = tf.keras.models.clone_model(model)
                logger.info("Optimized model for inference")
                
            except Exception as e:
                logger.warning(f"Could not set TensorFlow optimizations: {str(e)}")
                
        except Exception as e:
            logger.warning(f"Could not determine model input shape: {str(e)}")
            
except Exception as e:
    logger.error(f"❌ Error in model loading process: {str(e)}")
    model = None

# Perform a warmup prediction to initialize the model
if model is not None:
    try:
        logger.info("Performing warmup prediction...")
        # Create a dummy input tensor with the right shape
        dummy_input = np.zeros((1, target_height, target_width, 3), dtype=np.float32)
        # Run prediction with small timeout
        _ = model.predict(dummy_input, batch_size=1, verbose=0)
        logger.info("✓ Warmup prediction completed")
    except Exception as e:
        logger.warning(f"Warmup prediction failed: {str(e)}")

# Define image preprocessing for Keras model - MODIFIED to use the correct target size
def preprocess_image(img, target_size=(128, 128)):
    """Preprocess image for Keras model prediction"""
    # Use the dynamically determined target size if available
    if target_height and target_width:
        target_size = (target_height, target_width)
    
    logger.info(f"Preprocessing image to size: {target_size}")
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
        'classes': class_names,
        'input_shape': f"({target_height}, {target_width}, 3)" if model is not None else "unknown"
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

def predict_with_timeout(img_processed, timeout=30):
    """Run model prediction with timeout"""
    result_queue = queue.Queue()
    
    def prediction_worker():
        try:
            result = model.predict(img_processed, batch_size=1, verbose=0)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", e))
    
    # Start prediction in a separate thread
    worker = threading.Thread(target=prediction_worker)
    worker.daemon = True
    worker.start()
    
    # Wait for result with timeout
    worker.join(timeout=timeout)
    
    if worker.is_alive():
        return False, "Prediction timeout - model took too long to respond"
    
    try:
        status, result = result_queue.get(block=False)
        if status == "error":
            return False, str(result)
        return True, result
    except queue.Empty:
        return False, "No result returned from prediction thread"

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
        
        # Run inference with timeout
        logger.info(f"Running inference on image with shape {img_processed.shape}")
        success, result = predict_with_timeout(img_processed, timeout=45)  # 45 second timeout
        
        if not success:
            logger.error(f"❌ Prediction failed or timed out: {result}")
            return jsonify({'error': result}), 500
            
        predictions = result
        
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
            
    except ValueError as ve:
        # Handle shape mismatch errors specifically
        error_msg = str(ve)
        if "expected shape" in error_msg and "found shape" in error_msg:
            logger.error(f"❌ Shape mismatch error: {error_msg}")
            # Extract expected shape from error message
            import re
            expected_shape_match = re.search(r'expected shape=\(None, (\d+), (\d+), 3\)', error_msg)
            if expected_shape_match:
                h, w = expected_shape_match.groups()
                logger.info(f"Extracted expected dimensions: {h}x{w}")
                return jsonify({
                    'error': f"Image size mismatch. The model expects {h}x{w} images. Please try again with correct preprocessing.",
                    'expected_shape': f"{h}x{w}x3"
                }), 400
        
        # If not a shape error or couldn't extract dimensions
        logger.error(f"❌ ValueError in prediction: {error_msg}", exc_info=True)
        return jsonify({'error': error_msg}), 500
        
    except Exception as e:
        logger.error(f"❌ Error in prediction: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/predict/predict', methods=['POST'])
def predict_alias():
    return predict()      

if __name__ == '__main__':
    # Make sure to use the PORT environment variable provided by Render
    port = int(os.environ.get("PORT", 10000))
    
    # Use Gunicorn with specific worker configurations if available
    if 'gunicorn' in sys.modules:
        print(f"🚀 Starting with Gunicorn worker configurations...")
        # These settings will be picked up by Gunicorn
        os.environ['GUNICORN_CMD_ARGS'] = f"--timeout 120 --workers 1 --bind 0.0.0.0:{port} --preload"
        # Let Gunicorn handle the rest
    else:
        print(f"🚀 Starting Flask development server on port {port}...")
        # This is critical: Make sure to bind to 0.0.0.0 for Render to detect the port
        app.run(host='0.0.0.0', port=port)
