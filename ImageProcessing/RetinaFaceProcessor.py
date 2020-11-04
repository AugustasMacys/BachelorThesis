import cv2
import insightface
import numpy as np

from PIL import Image


img = Image.open(r"D:\deepfakes\extracted_images\apedduehoy_0.jpeg")
img_array = np.asarray(img)

model = insightface.model_zoo.get_model('retinaface_r50_v1')
model.prepare(ctx_id=-1, nms=0.4) # Ctx stands for GPU usage (could not install Might be worth looking in how to install)
bbox, landmark = model.detect(img_array, threshold=0.5, scale=1.0)

print(bbox)
print(landmark)

