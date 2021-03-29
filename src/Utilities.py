import json
import os
import pandas as pd
import shutil

from glob import glob

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_FILENAME = "metadata.json"

SRC_DIR = os.path.join(ROOT_DIR, "src")

DATAFRAMES_DIRECTORY = os.path.join(ROOT_DIR, "dataframes")
PAIR_DATAFRAMES_UPDATED_DIRECTORY = os.path.join(DATAFRAMES_DIRECTORY, "dataframes_pairs_updated")
PAIR_REAL_DATAFRAME = os.path.join(PAIR_DATAFRAMES_UPDATED_DIRECTORY, "real_crops.csv")
PAIR_FAKE_DATAFRAME = os.path.join(PAIR_DATAFRAMES_UPDATED_DIRECTORY, "fake_crops.csv")

MODELS_DIECTORY = os.path.join(ROOT_DIR, "trained_models")

VALIDATION_DATAFRAME_PATH = os.path.join(DATAFRAMES_DIRECTORY, "faces_validation_new.csv")
VALIDATION_DIRECTORY = os.path.join(ROOT_DIR, "data", "test")
VALIDATION_FACES_DIRECTORY = os.path.join(ROOT_DIR, "validation_faces")
VALIDATION_LABELS = os.path.join(VALIDATION_DIRECTORY, "labels.csv")

TRAIN_DIRECTORY = os.path.join(ROOT_DIR, "data", "train")
TRAIN_FAKE_FACES_DIRECTORY = os.path.join(ROOT_DIR, "training_fake_faces")
TRAIN_REAL_FACES_DIRECTORY = os.path.join(ROOT_DIR, "training_real_faces")

RESNET_FOLDER = os.path.join(ROOT_DIR, "special_models", "resnet_model")

MASKS_FOLDER = os.path.join(ROOT_DIR, "mask_xray_fake_new")
MASK_REAL_PATH = os.path.join(ROOT_DIR, "mask_xray_real", "black_mask.png")

HRNET_CONFIG_FILE = os.path.join(SRC_DIR, "xray_config", "hrnet_seg.yaml")

PRIVATE_TESTING_DIRECTORY = os.path.join(ROOT_DIR, "aws")
PRIVATE_TESTING_LABELS_PATH = os.path.join(PRIVATE_TESTING_DIRECTORY, "labels_updated.csv")

SEQUENCE_DATAFRAMES_FOLDER = os.path.join(DATAFRAMES_DIRECTORY, "dataframes_3dcnn")
SEQUENCE_DATAFRAME_PATH = os.path.join(SEQUENCE_DATAFRAMES_FOLDER, "3dcnn_dataframe.csv")
SEQUENCE_DATAFRAME_TESTING_PATH = os.path.join(SEQUENCE_DATAFRAMES_FOLDER, "3dtesting_dataframe.csv")

REAL_FOLDER_TO_IDENTIFIERS_PATH = os.path.join(ROOT_DIR, "real_folder_to_identifiers.pkl")
FAKE_FOLDER_TO_IDENTIFIERS_PATH = os.path.join(ROOT_DIR, "fake_folder_to_identifiers.pkl")
TESTING_FOLDER_TO_IDENTIFIERS_PATH = os.path.join(ROOT_DIR, "testing_folder_to_identifiers.pkl")

COVIAR_DATAFRAME_FOLDER = os.path.join(DATAFRAMES_DIRECTORY, "coviar_dataframes")
COVIAR_DATAFRAME_PATH = os.path.join(COVIAR_DATAFRAME_FOLDER, "coviar_dataframe.csv")
COVIAR_TEST_DATAFRAME_PATH = os.path.join(COVIAR_DATAFRAME_FOLDER, "coviar_test_dataframe.csv")

FACE_XRAY_REAL_DATAFRAME_PATH = os.path.join(PAIR_DATAFRAMES_UPDATED_DIRECTORY, "real_pairs_xray.csv")
FACE_XRAY_FAKE_DATAFRAME_PATH = os.path.join(PAIR_DATAFRAMES_UPDATED_DIRECTORY, "fake_pairs_xray.csv")

PREVIEW_MODELS = os.path.join(ROOT_DIR, "preview", "trained_models")
PREVIEW_DATAFRAMES = os.path.join(ROOT_DIR, "preview", "dataframes")
PREVIEW_TEST = os.path.join(PREVIEW_DATAFRAMES, "test_dataframe.csv")
PREVIEW_TRAIN = os.path.join(PREVIEW_DATAFRAMES, "training_dataframe.csv")


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


def get_number_of_real_videos_in_folders():
    directories_to_traverse = glob(os.path.join(TRAIN_DIRECTORY, '*', '*') + r'\\')
    list_real_in_folders = []
    for directory in directories_to_traverse:
        path = os.path.join(directory, METADATA_FILENAME)
        current_real = 0
        with open(path) as f:
            data = json.load(f)
            for id in data:
                if data[id]["label"] == "REAL":
                    current_real += 1

        list_real_in_folders.append(current_real)

    return list_real_in_folders


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
