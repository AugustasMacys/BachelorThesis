import argparse
import cv2
import insightface
import ntpath
import numpy as np
import pandas as pd
import os

from tqdm import tqdm


from utilities import TRAIN_FACES_DIRECTORY, DATAFRAMES_DIRECTORY, REAL_VIDEO_SAMPLE_DIRECTORY,\
    TEST_VIDEO_READER_DIRETORY

FACE_SIZE = 320


def video_frame_extractor(video_name, folder):
    capturator = cv2.VideoCapture(video_name)
    frames_number = int(capturator.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capturator.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = 1.0
    for i in range(frames_number):
        capturator.grab()
        if i % 20 == 0: # Depends how many frames I want
            success, frame = capturator.retrieve()

            if not success:
                continue

            if width <= 300:
                scale = 2.0
            elif 1000 < width <= 1900:
                scale = 0.5
            elif width > 1900:
                scale = 0.33

            bounding_box, _ = model.detect(frame, threshold=0.7, scale=scale)
            if bounding_box.shape[0] == 0:
                continue

            x_min = bounding_box[:, 0]
            y_min = bounding_box[:, 1]
            x_max = bounding_box[:, 2]
            y_max = bounding_box[:, 3]

            areas = (y_max - y_min) * (x_max - x_min)
            max_face_idx = areas.argmax()

            w = x_max[max_face_idx] - x_min[max_face_idx]
            h = y_max[max_face_idx] - y_min[max_face_idx]

            margin_width = w // 3
            margin_height = h // 3

            frame = frame[max(int(y_min[max_face_idx] - margin_height), 0):int(y_max[max_face_idx] + margin_height),
                    max(int(x_min[max_face_idx] - margin_width), 0):int(x_max[max_face_idx] + margin_width)]

            identifier = ntpath.basename(video_name)[:-4] + '_' + str(i)
            save_string = os.path.join(folder, identifier) + ".png"
            cv2.imwrite(save_string, frame)

    capturator.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoFolder")
    args = parser.parse_args()

    training_dataframe_path = os.path.join(DATAFRAMES_DIRECTORY, "training_dataframe.csv")
    video_filenames = pd.read_csv(training_dataframe_path)["video_name"]
    # video_filenames = [os.path.join(REAL_VIDEO_SAMPLE_DIRECTORY, path) for path in os.listdir(REAL_VIDEO_SAMPLE_DIRECTORY)]
    # print(video_filenames)
    # output_folder = TEST_VIDEO_READER_DIRETORY
    output_folder = TRAIN_FACES_DIRECTORY
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=0, nms=0.4)
    with tqdm(total=len(video_filenames)) as bar:
        for video in video_filenames:
            video_frame_extractor(video, output_folder)
            bar.update()
