import json
import os
import pandas as pd
import shutil

from glob import glob

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_FILENAME = "metadata.json"

FACES_DIRECTORY = os.path.join(ROOT_DIR, "extracted_images")
SAMPLE_VIDEO_DIRECTORY = os.path.join(ROOT_DIR, "video_examples")
REAL_VIDEO_SAMPLE_DIRECTORY = os.path.join(ROOT_DIR, "real_video_examples")
TEST_VIDEO_READER_DIRETORY = os.path.join(ROOT_DIR, "test_video_reader")
DATAFRAMES_DIRECTORY = os.path.join(ROOT_DIR, "dataframes")
NOISY_STUDENT_DIRECTORY = os.path.join(ROOT_DIR, "noisy_student_weights")
MODELS_DIECTORY = os.path.join(ROOT_DIR, "trained_models")
VALIDATION_DIRECTORY = os.path.join(ROOT_DIR, "data", "test")
VALIDATION_FACES_DIRECTORY = os.path.join(ROOT_DIR, "validation_faces")
VALIDATION_LABELS = os.path.join(VALIDATION_DIRECTORY, "labels.csv")
TRAIN_DIRECTORY = os.path.join(ROOT_DIR, "data", "train")
TRAIN_FACES_DIRECTORY = os.path.join(ROOT_DIR, "initial_training_faces")

VALIDATION_DATAFRAME_PATH = os.path.join(DATAFRAMES_DIRECTORY, "faces_validation.csv")
TRAINING_DATAFRAME_PATH = os.path.join(DATAFRAMES_DIRECTORY, "faces_training.csv")


EFFICIENT_B0_SIZE = 224
EFFICIENT_B4_SIZE = 320


def get_specific_video_names(folder, number=20, label='FAKE'):
    path = os.path.join(folder, METADATA_FILENAME)
    names = []
    with open(path) as f:
        data = json.load(f)
        for id in data:
            if data[id]["label"] == label:
                names.append(id)

            if len(names) == number:
                return names

    return names


def get_fake_videos_with_corresponding_original_videos(folder, original_limit=125,
                                                       fake_limit=200):
    path = os.path.join(folder, METADATA_FILENAME)
    fake_list = list()
    original_set = set()
    original_limit = original_limit
    with open(path) as f:
        data = json.load(f)
        for id in data:
            if data[id]["label"] == "FAKE":
                if len(fake_list) < fake_limit:
                    fake_list.append(os.path.join(folder, id))
                if original_limit > len(original_set):
                    if data[id]["original"]:
                        original_set.add(os.path.join(folder, data[id]["original"]))

            if len(fake_list) == fake_limit and len(original_set) == original_limit:
                return fake_list, original_set

    return fake_list, original_set


def get_original_videos_of_fake_videos(fake_videos, metadata_folder):
    metadata_path = os.path.join(metadata_folder, METADATA_FILENAME)
    videos_to_fakes = dict()
    with open(metadata_path) as f:
        data = json.load(f)
        for video in fake_videos:
            if data[video]["original"]:
                videos_to_fakes[video] = data[video]["original"]

    return videos_to_fakes


def copy_specific_videos(src_folder, src_names, destination_folder):
    for name in src_names:
        path = os.path.join(src_folder, name)
        destination_path = os.path.join(destination_folder, name)
        shutil.copyfile(path, destination_path)


def write_list_to_file(list, dest, label):
    path = os.path.join(dest, label)
    with open(path, 'w') as f:
        for name in list:
            f.write("%s\n" % name)


def read_txt_as_list(filename):
    with open(filename) as f:
        lines = f.read().splitlines()

    return lines


def list_to_dataframe(video_names, column_name="videoname"):
    return pd.DataFrame(video_names, columns=[column_name])


def get_all_files_with_extension_from_folder(folder, extension):
    video_files = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            video_files.append(file)

    return video_files


if __name__ == '__main__':
    directories_to_traverse = glob(os.path.join(TRAIN_DIRECTORY, '*', '*') + r'\\')
    all_fake_videos = []
    all_real_videos = []
    for folder in directories_to_traverse:
        current_folder_fake_video_names, current_folder_real_video_names = get_fake_videos_with_corresponding_original_videos(folder)
        all_fake_videos.append(current_folder_fake_video_names)
        all_real_videos.append(current_folder_real_video_names)

    len_lists_fakes = [len(item) for item in all_fake_videos]
    len_lists_real = [len(item) for item in all_real_videos]

    with open('lenFakes.txt', 'w') as f:
        for item in len_lists_fakes:
            f.write("%s\n" % item)

    with open('lenReal.txt', 'w') as f:
        for item in len_lists_real:
            f.write("%s\n" % item)

    flat_fake = [item for sublist in all_fake_videos for item in sublist]
    flat_real = [item for sublist in all_real_videos for item in sublist]

    fake_dataframe = pd.DataFrame(flat_fake, columns=["video_name"])
    real_dataframe = pd.DataFrame(flat_real, columns=["video_name"])

    fake_dataframe["label"] = 1
    real_dataframe["label"] = 0

    final_dataframe = pd.concat([fake_dataframe, real_dataframe])
    final_dataframe.to_csv(os.path.join(DATAFRAMES_DIRECTORY, "training_dataframe.csv"), index=False)
