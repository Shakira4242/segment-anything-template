import cv2
import base64
import banana_dev as banana
import numpy as np


image = cv2.imread('girl.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

out = banana.run("9f9ce478-e635-439c-b701-a2a6ba6244d3","6256dbd2-7f1e-4264-bf9e-ecc463f8c275",{"image":image_string})

print(out)