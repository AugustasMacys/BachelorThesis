import cv2
import os
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from tqdm import tqdm


import face_alignment
from scipy.spatial import ConvexHull
import torch


from src.Utilities import MASKS_FOLDER, PAIR_REAL_DATAFRAME


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


def create_mask(path):
    img = Image.open(path)
    img = np.asarray(img)
    img = isotropically_resize_image(img, 224)
    height, width, _ = img.shape
    black_mask = np.zeros((height, width))
    preds = fa.get_landmarks(img)
    if preds is None:
        return path
    if len(preds) < 1:
        return path
    points = preds[0]
    hull = ConvexHull(points)
    del preds
    img = cv2.drawContours(black_mask, [points[hull.vertices].astype(int)], -1, (1, 1, 1), thickness=cv2.FILLED)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = 4.0 * np.multiply(img, (1.0 - img))
    img = (img * 255).astype(np.uint8)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    folder = path.split("\\")[3]
    identifier = path.split("\\")[4]
    directory_to_create = os.path.join(MASKS_FOLDER, folder)
    if not os.path.exists(directory_to_create):
        os.mkdir(directory_to_create)

    save_string = os.path.join(directory_to_create, identifier)
    cv2.imwrite(save_string, img)
    return ""


if __name__ == '__main__':
    torch.cuda.empty_cache()
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                      flip_input=False)


    # According to research paper, we create masks from real images
    image_paths = pd.unique(pd.read_csv(PAIR_REAL_DATAFRAME).image_path)

    files_without_mask = []
    with tqdm(total=len(image_paths)) as bar:
        for path in image_paths:
            path = create_mask(path)
            if path != "":
                files_without_mask.append(path)
                with open('files_without_mask', 'wb') as fp:
                    pickle.dump(files_without_mask, fp)
            bar.update()
