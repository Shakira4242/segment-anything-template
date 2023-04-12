import cv2
import base64
import banana_dev as banana
import numpy as np


image = cv2.imread('banana.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

out = banana.run("api_key","model_key",{"image":image_string})
masks = out["modelOutputs"][0]["masks"]

for mask in masks:
    mask["segmentation"] = np.array(mask["segmentation"])