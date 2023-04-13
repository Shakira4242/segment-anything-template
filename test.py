import cv2
import base64
import banana_dev as banana
import numpy as np


image = cv2.imread('girl.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

out = banana.run("9f9ce478-e635-439c-b701-a2a6ba6244d3","55c1d7a4-6076-44a9-9c1d-15f3f891cc3f",{"image":image_string})

print(out)