import json
import os
import shutil

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
    for name in src_names:
        path = os.path.join(src_folder, name)
        destination_path = os.path.join(destination_folder, name)
        shutil.copyfile(path, destination_path)


def write_list_to_file(list, dest, label):
    path = os.path.join(dest, label)
    with open(path, 'w') as f:
        for name in list:
            f.write("%s\n" % name)


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
