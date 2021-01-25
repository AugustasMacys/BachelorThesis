from PIL import Image
import numpy as np
import cv2
from scipy.spatial import ConvexHull
import face_alignment
import pandas as pd
from tqdm import tqdm
import os


from utilities import MASKS_FOLDER, PAIR_UPDATED_REAL_DATAFRAME


def create_mask(path):
    img = Image.open(path)
    img = np.asarray(img)
    height, width, _ = img.shape
    black_mask = np.zeros((height, width))
    preds = fa.get_landmarks(img)
    if len(preds) < 1:
        return
    points = preds[0]
    hull = ConvexHull(points)
    del preds
    img = cv2.drawContours(black_mask, [points[hull.vertices].astype(int)], -1, (1, 1, 1), thickness=cv2.FILLED)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = 4.0 * np.multiply(img, (1.0 - img))
    img = (img * 255).astype(int)
    folder = path.split("\\")[3]
    identifier = path.split("\\")[4]
    directory_to_create = os.path.join(MASKS_FOLDER, folder)
    if not os.path.exists(directory_to_create):
        os.mkdir(directory_to_create)

    save_string = os.path.join(directory_to_create, identifier)
    cv2.imwrite(save_string, img)


if __name__ == '__main__':
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    # According to research paper, we create masks from real images
    image_paths = pd.read_csv(PAIR_UPDATED_REAL_DATAFRAME).image_path

    with tqdm(total=len(image_paths)) as bar:
        for path in image_paths:
            create_mask(path)
            bar.update()
