import torch
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import base64
def init():
    global model
    sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
    sam.to("cuda")
    model = SamPredictor(sam)

def inference(model_inputs):
    global model
    image_string = model_inputs.get('image', None)
    if image_string == None:
        return {'message': "No image provided"}

    img_original = base64.b64decode(image_string)
    img_as_np = np.frombuffer(img_original, dtype=np.uint8)
    image = cv2.imdecode(img_as_np, flags=1)
    model.set_image(image)
    image_embedding = model.get_image_embedding().cpu().numpy()
    return {"embedding": image_embedding}