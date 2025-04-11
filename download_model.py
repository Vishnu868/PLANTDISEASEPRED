import gdown
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL to your Keras model
url = 'https://drive.google.com/uc?id=1Ny1hs9hxOsrH5fBU4kqvutfj-vlsCDk5'
output = 'model.h5'  # Use .h5 extension for better compatibility

def download_model():
    logger.info(f"Downloading model from {url} to {output}")
    try:
        # Force download even if file exists
        gdown.download(url, output, quiet=False, fuzzy=True)
        
        # Verify file downloaded successfully
        if os.path.exists(output) and os.path.getsize(output) > 0:
            logger.info(f"✅ Model downloaded successfully: {os.path.getsize(output)} bytes")
            # Create best.pt as a copy if you want to keep that filename
            try:
                import shutil
                shutil.copy(output, 'best.pt')
                logger.info("Created copy as best.pt")
                return True
            except Exception as e:
                logger.error(f"Could not create copy as best.pt: {str(e)}")
        else:
            logger.error("❌ Model download failed or file is empty")
            return False
    except Exception as e:
        logger.error(f"❌ Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_model()
    if not success:
        sys.exit(1)  # Exit with error code to signal failure to the build script
