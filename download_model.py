import gdown

url = 'https://drive.google.com/uc?id=1eWnEpZ3Yxny9MaQgtjYsBTKJxWczPP9J'
output = 'best.pt'

gdown.download(url, output, quiet=False)
