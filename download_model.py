# download_model.py
import gdown

url = 'https://drive.google.com/uc?id=1oPMxoHAXVAQNcNPa9EUKCXx7g4sqaKsA'
output = 'best.pt'
gdown.download(url, output, quiet=False)
