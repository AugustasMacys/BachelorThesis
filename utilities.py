import json
import os
import pandas as pd
import shutil
import ntpath

from glob import glob

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_FILENAME = "metadata.json"

FACES_DIRECTORY = os.path.join(ROOT_DIR, "extracted_images")
SAMPLE_VIDEO_DIRECTORY = os.path.join(ROOT_DIR, "video_examples")
REAL_VIDEO_SAMPLE_DIRECTORY = os.path.join(ROOT_DIR, "real_video_examples")
TEST_VIDEO_READER_DIRETORY = os.path.join(ROOT_DIR, "test_video_reader")
DATAFRAMES_DIRECTORY = os.path.join(ROOT_DIR, "dataframes")
PAIR_DATAFRAMES_DIRECTORY = os.path.join(ROOT_DIR, "dataframes_pairs")
PAIR_REAL_DATAFRAME = os.path.join(PAIR_DATAFRAMES_DIRECTORY, "real_crops.csv")
PAIR_FAKE_DATAFRAME = os.path.join(PAIR_DATAFRAMES_DIRECTORY, "fake_crops.csv")
NOISY_STUDENT_DIRECTORY = os.path.join(ROOT_DIR, "noisy_student_weights")
MODELS_DIECTORY = os.path.join(ROOT_DIR, "trained_models")
VALIDATION_DIRECTORY = os.path.join(ROOT_DIR, "data", "test")
VALIDATION_FACES_DIRECTORY = os.path.join(ROOT_DIR, "validation_faces")
VALIDATION_LABELS = os.path.join(VALIDATION_DIRECTORY, "labels.csv")
TRAIN_DIRECTORY = os.path.join(ROOT_DIR, "data", "train")
TRAIN_FACES_DIRECTORY = os.path.join(ROOT_DIR, "initial_training_faces")
TRAIN_FAKE_FACES_DIRECTORY = os.path.join(ROOT_DIR, "training_fake_faces")
TRAIN_REAL_FACES_DIRECTORY = os.path.join(ROOT_DIR, "training_real_faces")
RESNET_FOLDER = os.path.join(ROOT_DIR, "resnet_model")

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


def get_fake_videos_with_corresponding_original_videos(folder, original_limit,
                                                       fake_limit):
    path = os.path.join(folder, METADATA_FILENAME)
    fake_list = list()
    original_list = list()
    original_limit = original_limit
    with open(path) as f:
        data = json.load(f)
        for id in data:
            if data[id]["label"] == "FAKE":
                if data[id]["original"]:
                    original_full_path = os.path.join(folder, data[id]["original"])
                    if original_full_path not in original_list:
                        original_list.append(os.path.join(folder, data[id]["original"]))
                        fake_list.append(os.path.join(folder, id))

            if len(fake_list) == fake_limit and len(original_list) == original_limit:
                return fake_list, original_list

    return fake_list, original_list


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


if __name__ == '__main__':
    real_videos_in_folder = get_number_of_real_videos_in_folders()
    directories_to_traverse = glob(os.path.join(TRAIN_DIRECTORY, '*', '*') + r'\\')
    all_fake_videos = []
    all_real_videos = []
    index = 0
    for folder in directories_to_traverse:
        needed_real_videos = real_videos_in_folder[index]
        current_folder_fake_video_names, current_folder_real_video_names = get_fake_videos_with_corresponding_original_videos(folder,
                                                                                                                              needed_real_videos, needed_real_videos)
        index += 1
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

    # dataframe_dictionary = {
    #     "fake_video_name": flat_fake,
    #     "real_video_name": flat_real
    # }
    #
    # images_dataframe = pd.DataFrame(dataframe_dictionary)
    # images_dataframe = pd.DataFrame(images_dataframe.to_numpy().flatten())

    # images_dataframe.rename({'0': "video_name"})
    # images_dataframe["label"] = 0

    fake_dataframe = pd.DataFrame(flat_fake, columns=["video_name"])
    real_dataframe = pd.DataFrame(flat_real, columns=["video_name"])

    fake_dataframe["label"] = 1
    real_dataframe["label"] = 0
    #
    # final_dataframe = pd.concat([fake_dataframe, real_dataframe], ignore_index=True)
    real_dataframe.to_csv(os.path.join(PAIR_DATAFRAMES_DIRECTORY, "real_training_dataframe.csv"), index=False)
    fake_dataframe.to_csv(os.path.join(PAIR_DATAFRAMES_DIRECTORY, "fake_training_dataframe.csv"), index=False)

    # fake_frame_faces = glob(os.path.join(TRAIN_FAKE_FACES_DIRECTORY, '*'))
    # real_frame_faces = glob(os.path.join(TRAIN_REAL_FACES_DIRECTORY, '*'))
    #
    # training_faces_real = pd.DataFrame(real_frame_faces, columns=["image_path"])
    # training_faces_fake = pd.DataFrame(fake_frame_faces, columns=["image_path"])
    #
    # training_faces_real["label"] = 0
    # training_faces_fake["label"] = 1
    #
    # training_faces_dataframe = pd.concat([training_faces_real, training_faces_fake], ignore_index=True)
    # training_faces_dataframe.to_csv(TRAINING_DATAFRAME_PATH, index=False)
