import cv2
import insightface
import numpy as np

from PIL import Image
from skimage.transform import SimilarityTransform


def norm_crop(img, landmark, image_size=112):
    ARCFACE_SRC = np.array([[
        [122.5, 141.25],
        [197.5, 141.25],
        [160.0, 178.75],
        [137.5, 225.25],
        [182.5, 225.25]
    ]], dtype=np.float32)

    def estimate_norm(lmk):
        assert lmk.shape == (5, 2)

        tform = SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = np.inf
        src = ARCFACE_SRC

        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]

        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

        return min_M, min_index

    M, pose_index = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


# img = Image.open(r"D:\deepfakes\extracted_images\apedduehoy_0.jpeg")
img = cv2.imread(r"D:\deepfakes\extracted_images\apedduehoy_0.jpeg")
# print(img)
img_array = np.asarray(img)

model = insightface.model_zoo.get_model('retinaface_r50_v1')
model.prepare(ctx_id=-1, nms=0.4) # Ctx stands for GPU usage (could not install Might be worth looking in how to install)
bbox, landmark = model.detect(img_array, threshold=0.5, scale=1.0)
# areas = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
# max_face_idx = areas.argmax()
# landm = landmark[max_face_idx]
# landmarks = landm.reshape(5, 2).astype(np.int)
# img = norm_crop(img_array, landmarks, image_size=380)
# aligned = Image.fromarray(img[:, :, ::-1])
# aligned.save(r"D:\deepfakes\1.jpg")
bbox = bbox[0]
x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

# crop = img.crop((y, y+h, x, x+w))
img = img[int(y):int(y+h), int(x):int(x+w)]
cv2.imwrite(r"D:\deepfakes\2.jpg", img)

print(bbox)

