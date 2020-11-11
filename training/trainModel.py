import os
import numpy as np

from utilities import get_original_videos_of_fake_videos,\
    list_to_dataframe, read_txt_as_list
from utilities import FACES_DIRECTORY, ROOT_DIR, SAMPLE_VIDEO_DIRECTORY

IMAGE_SIZE = 224
BATCH_SIZE = 64

FAKE = "FAKES.TXT"
FAKE_VIDEOS = os.path.join(SAMPLE_VIDEO_DIRECTORY, FAKE)

REAL = "REAL.TXT"
REAL_VIDEOS = os.path.join(SAMPLE_VIDEO_DIRECTORY, REAL)

real_videos = read_txt_as_list(REAL_VIDEOS)
fake_videos = read_txt_as_list(FAKE_VIDEOS)

real_dataframe = list_to_dataframe(real_videos)
fake_dataframe = list_to_dataframe(fake_videos)

real_dataframe["label"] = 0
fake_dataframe["label"] = 1

real_dataframe["original"] = np.nan

metadata_folder = os.path.join(ROOT_DIR, "data", "train", "train_00", "dfdc_train_part_0")
fake_to_original = get_original_videos_of_fake_videos(fake_videos, metadata_folder)

fake_dataframe["original"] = fake_dataframe["videoname"].map(fake_to_original)

print(fake_to_original)
print(fake_dataframe)


