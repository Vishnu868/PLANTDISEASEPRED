import gdown

url = 'https://drive.google.com/uc?id=1eWnEpZ3Yxny9MaQgtjYsBTKJxWczPP9J'
output = 'best.pt'  # Keep the original name as you requested
gdown.download(url, output, quiet=False)
