import requests, tempfile, os

url = "http://127.0.0.1:8000/dehaze"
files = {'img': ('111.jpg', open('111.jpg', 'rb'), 'image/jpg', {})}
res = requests.request("POST", url, data={'type':'1'}, files=files)
tf = tempfile.NamedTemporaryFile(prefix='haze', dir='received_haze_images').name
print(res.text, tf)
