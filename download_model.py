import gdown
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL to your Keras model
url = "https://drive.google.com/uc?id=1mQHatbcqWQc0HY7r9mJa1ahosUH6AO6M"
output = 'trained_model.h5'  # Use .h5 extension instead of .keras for better compatibility

logger.info(f"Downloading model from {url} to {output}")
try:
    gdown.download(url, output, quiet=False)
    
    # Verify file downloaded successfully
    if os.path.exists(output) and os.path.getsize(output) > 0:
        logger.info(f"✅ Model downloaded successfully: {os.path.getsize(output)} bytes")
        # Create best.pt as a copy if you want to keep that filename
        try:
            import shutil
            shutil.copy(output, 'best.pt')
            logger.info("Created copy as best.pt")
        except Exception as e:
            logger.warning(f"Could not create copy as best.pt: {str(e)}")
    else:
        logger.error("❌ Model download failed or file is empty")
except Exception as e:
    logger.error(f"❌ Error downloading model: {str(e)}")
