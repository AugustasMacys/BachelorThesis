import json
import os
import pandas as pd
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_FILENAME = "metadata.json"

FACES_DIRECTORY = os.path.join(ROOT_DIR, "extracted_images")
SAMPLE_VIDEO_DIRECTORY = os.path.join(ROOT_DIR, "video_examples")


def get_specific_video_names(folder, number=20, label='FAKE'):
    path = os.path.join(folder, METADATA_FILENAME)
    print(path)
    names = []
    with open(path) as f:
        data = json.load(f)
        for id in data:
            if data[id]["label"] == label:
                names.append(id)

            if len(names) == number:
                return names

    return names


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


def get_all_videos_from_folder(folder):
    video_files = []
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            video_files.append(os.path.join(folder, file))

    return video_files



# destination_folder = r"D:\deepfakes\video_examples"
# folder00 = r"D:\deepfakes\data\train\train_00\dfdc_train_part_0"
# fakes = get_specific_video_names(folder00)
# real = get_specific_video_names(folder00, label="REAL")
#
# write_list_to_file(fakes, destination_folder, "FAKES.txt")
# write_list_to_file(real, destination_folder, "REAL.txt")
#
# copy_specific_videos(folder00, fakes, destination_folder)
# copy_specific_videos(folder00, real, destination_folder)
