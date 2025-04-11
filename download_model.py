import gdown
import os
import logging
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL to your Keras model (ensure this is the correct Google Drive share link)
url = 'https://drive.google.com/uc?id=1Ny1hs9hxOsrH5fBU4kqvutfj-vlsCDk5'
output = 'model.keras'  # Changed to proper extension

logger.info(f"Downloading model from {url} to {output}")
try:
    gdown.download(url, output, quiet=False)
    
    # Verify file downloaded successfully
    if os.path.exists(output) and os.path.getsize(output) > 0:
        logger.info(f"✅ Model downloaded successfully: {os.path.getsize(output)} bytes")
        
        # Try to load the model to verify it's valid
        try:
            logger.info("Attempting to verify model format...")
            model = tf.keras.models.load_model(output)
            logger.info(f"✅ Model verified! Output shape: {model.output.shape}")
            
            # Create symbolic link if you really want to keep best.pt name
            try:
                if os.path.exists('best.pt'):
                    os.remove('best.pt')
                os.symlink(output, 'best.pt')
                logger.info("Created symbolic link best.pt -> model.keras")
            except Exception as e:
                logger.warning(f"Could not create symbolic link: {str(e)}")
                
        except Exception as e:
            logger.error(f"❌ Model format verification failed: {str(e)}")
    else:
        logger.error("❌ Model download failed or file is empty")
except Exception as e:
    logger.error(f"❌ Error downloading model: {str(e)}")
