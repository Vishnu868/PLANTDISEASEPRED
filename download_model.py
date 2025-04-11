import gdown
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    url = 'https://drive.google.com/uc?id=1eWnEpZ3Yxny9MaQgtjYsBTKJxWczPP9J'
    output = 'best.pt'
    
    # Check if the file already exists
    if os.path.exists(output):
        file_size = os.path.getsize(output)
        logger.info(f"Model file already exists ({file_size} bytes). Skipping download.")
        return True
    
    try:
        logger.info(f"Downloading model from Google Drive...")
        gdown.download(url, output, quiet=False)
        
        # Verify the file was downloaded
        if os.path.exists(output):
            file_size = os.path.getsize(output)
            logger.info(f"Model downloaded successfully ({file_size} bytes)")
            return True
        else:
            logger.error("Model download failed - file not found")
            return False
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    download_model()
