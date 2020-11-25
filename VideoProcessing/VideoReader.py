import argparse
import cv2
import insightface
import ntpath
import numpy as np
import pandas as pd
import os

from tqdm import tqdm

from skimage.transform import SimilarityTransform
from insightface.utils.face_align import arcface_src

from utilities import TRAIN_FACES_DIRECTORY, DATAFRAMES_DIRECTORY

FACE_SIZE = 224
INSIGHTFACE_SIZE = 112
FACTOR = FACE_SIZE / INSIGHTFACE_SIZE
ARCFACE_REFERENCE = arcface_src * FACTOR


# insightface/utils/face_align.py
def norm_crop(img, landmark, arcface_reference, image_size=112):
    def get_transform_matrix(landmarks):
        assert landmarks.shape == (5, 2)

        tform = SimilarityTransform()
        lmk_tran = np.insert(landmarks, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = np.inf
        src = arcface_reference

        for i in np.arange(src.shape[0]):
            tform.estimate(landmarks, src[i])
        transform_matrix = tform.params[0:2, :]

        results = np.dot(transform_matrix, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = transform_matrix
            min_index = i

        return min_M, min_index

    M, pose_index = get_transform_matrix(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


def video_frame_extractor(video_name, folder):
    capturator = cv2.VideoCapture(video_name)
    frames_number = int(capturator.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(frames_number):
        capturator.grab()
        if i % 20 == 0: # Depends how many frames I want
            success, frame = capturator.retrieve()
            if not success:
                continue

            bounding_box, landmarks = model.detect(frame, threshold=0.5, scale=1.0)
            if bounding_box.shape[0] == 0:
                continue

            areas = (bounding_box[:, 3] - bounding_box[:, 1]) * (bounding_box[:, 2] - bounding_box[:, 0])
            max_face_idx = areas.argmax()
            face_landmark = landmarks[max_face_idx]

            face_landmark = face_landmark.reshape(5, 2).astype(np.int)
            transformed_image = norm_crop(frame, face_landmark, ARCFACE_REFERENCE, image_size=224)

            identifier = ntpath.basename(video_name)[:-4] + '_' + str(i)
            save_string = os.path.join(folder, identifier) + ".png"
            cv2.imwrite(save_string, transformed_image)

    capturator.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoFolder")
    args = parser.parse_args()

    training_dataframe_path = os.path.join(DATAFRAMES_DIRECTORY, "training_dataframe.csv")
    video_filenames = pd.read_csv(training_dataframe_path)["video_name"]
    output_folder = TRAIN_FACES_DIRECTORY
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=0, nms=0.4)
    with tqdm(total=len(video_filenames)) as bar:
        for video in video_filenames:
            video_frame_extractor(video, output_folder)
            bar.update()
