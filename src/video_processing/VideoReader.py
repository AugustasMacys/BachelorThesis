import cv2
import insightface
import ntpath
import pandas as pd
import os
from glob import glob

from tqdm import tqdm


from src.Utilities import TRAIN_FAKE_FACES_DIRECTORY, TRAIN_REAL_FACES_DIRECTORY, DATAFRAMES_DIRECTORY, \
    VALIDATION_FACES_DIRECTORY, VALIDATION_DIRECTORY, PAIR_DATAFRAMES_DIRECTORY, TRAIN_FAKE_FACES_DIRECTORY2,\
    TRAIN_FAKE_FACES_DIRECTORY3, TRAIN_FAKE_FACES_DIRECTORY3_SSD

FACE_SIZE = 224


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

            margin_width = w // 4
            margin_height = h // 4

            frame = frame[max(int(y_min[max_face_idx] - margin_height), 0):int(y_max[max_face_idx] + margin_height),
                    max(int(x_min[max_face_idx] - margin_width), 0):int(x_max[max_face_idx] + margin_width)]

            identifier = ntpath.basename(video_name)[:-4] + '_' + str(i)
            directory_to_create = os.path.join(folder, ntpath.basename(video_name)[:-4])
            if not os.path.exists(directory_to_create):
                os.mkdir(directory_to_create)
            save_string = os.path.join(directory_to_create, identifier) + ".png"
            cv2.imwrite(save_string, frame)

    capturator.release()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--videoFolder")
    # args = parser.parse_args()
    # validation_video_paths = glob(os.path.join(VALIDATION_DIRECTORY, "*.mp4"))
    # training_real_dataframe_path = os.path.join(PAIR_DATAFRAMES_DIRECTORY, "real_training_dataframe.csv")
    training_fake_dataframe_path = os.path.join(PAIR_DATAFRAMES_DIRECTORY, "fake_training_dataframe3.csv")
    # real_video_filenames_dataframe = pd.read_csv(training_real_dataframe_path)
    fake_video_filenames_dataframe = pd.read_csv(training_fake_dataframe_path)
    # real_video_filenames = real_video_filenames_dataframe["video_name"].values
    fake_video_filenames_hdd = list(fake_video_filenames_dataframe["video_name"].values[:48000])
    fake_video_filenames_ssd = list(fake_video_filenames_dataframe["video_name"].values[48000:])
    # print(len(real_video_filenames))
    # print(len(fake_video_filenames))
    # print(len(real_video_filenames))
    # exit(0)
    # video_filenames = [os.path.join(REAL_VIDEO_SAMPLE_DIRECTORY, path) for path in os.listdir(REAL_VIDEO_SAMPLE_DIRECTORY)]
    # print(video_filenames)
    # output_folder = TEST_VIDEO_READER_DIRETORY
    # output_folder = TRAIN_REAL_FACES_DIRECTORY
    # output_folder = VALIDATION_FACES_DIRECTORY
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=0, nms=0.4)
    # with tqdm(total=len(real_video_filenames)) as bar:
    #     for video in real_video_filenames:
    #         video_frame_extractor(video, output_folder)
    #         bar.update()

    output_folder = TRAIN_FAKE_FACES_DIRECTORY3_SSD
    with tqdm(total=len(fake_video_filenames_ssd)) as bar:
        for video in fake_video_filenames_ssd:
            video_frame_extractor(video, output_folder)
            bar.update()

    output_folder = TRAIN_FAKE_FACES_DIRECTORY3
    with tqdm(total=len(fake_video_filenames_hdd)) as bar:
        for video in fake_video_filenames_hdd:
            video_frame_extractor(video, output_folder)
            bar.update()
