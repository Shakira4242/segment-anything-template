# 🍌 Banana Serverless Segment Anything Template

This repo gives a framework for serving Meta AI's SAM (segment anything) in production using simple HTTP servers.

To understand how the model works and how to use the outputs refer to the model git repo [here](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb)

Look at `test.py` for instructions on how to call this model on locally as well as deployed on banana.

```
import cv2
import base64
import banana_dev as banana
import numpy as np

#Open image
image = cv2.imread('banana.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Convert image to string
image_string = base64.b64encode(cv2.imencode('.jpg', image)[1]).decode()

#Call banana
out = banana.run("api_key","model_key",{"image":image_string})
masks = out["modelOutputs"][0]["masks"]

#Convert segmentation list to numpy array
for mask in masks:
    mask["segmentation"] = np.array(mask["segmentation"])

```