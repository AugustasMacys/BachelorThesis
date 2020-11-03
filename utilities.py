import json
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_FILENAME = "metadata.json"


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


def copy_specific_videos(src_folder, src_names, destination_folder):
    


folder00 = r"D:\deepfakes\data\train\train_00\dfdc_train_part_0"
print(get_specific_video_names(folder00))
