from PIL import Image
import numpy as np
import cv2
from scipy.spatial import ConvexHull
import face_alignment
import pandas as pd
from tqdm import tqdm


from utilities import PAIR_UPDATED_FAKE_DATAFRAME, PAIR_UPDATED_FAKE_DATAFRAME2, PAIR_UPDATED_FAKE_DATAFRAME3,\
    MASKS_FOLDER


def mask_creator(path):
    img = Image.open(path)
    img = np.asarray(img)
    height, width, _ = img.shape
    black_mask = np.zeros((height, width))
    preds = fa.get_landmarks(img)
    if len(preds) < 1:
        return
    points = preds[0]
    hull = ConvexHull(points)
    img = cv2.drawContours(black_mask, [points[hull.vertices].astype(int)], -1, (255, 255, 255), thickness=cv2.FILLED)
    cv2.imwrite(r"D:/deepfakes/x_ray_example_on_one_picture/convex_hull_black_frame.png", img)


if __name__ == '__main__':
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    # Masks are needed only for fake faces as real faces mask is just black
    image_paths = pd.read_csv(PAIR_UPDATED_FAKE_DATAFRAME).image_path

    with tqdm(total=len(image_paths)) as bar:
        for path in image_paths:
            mask_creator(path)
            bar.update()

    image_paths = pd.read_csv(PAIR_UPDATED_FAKE_DATAFRAME2).image_path
    with tqdm(total=len(image_paths)) as bar:
        for path in image_paths:
            mask_creator(path)
            bar.update()

    image_paths = pd.read_csv(PAIR_UPDATED_FAKE_DATAFRAME3).image_path
    with tqdm(total=len(image_paths)) as bar:
        for path in image_paths:
            mask_creator(path)
            bar.update()
