import cv2
import base64
import requests
image = cv2.imread('banana.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

res = requests.post("http://localhost:8000/",json={"image":string})

print(res.text)