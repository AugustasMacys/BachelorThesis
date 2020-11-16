import os
import numpy as np

import matplotlib.pyplot as plt

from training.trainUtilities import Unnormalize
from utilities import get_original_videos_of_fake_videos,\
    list_to_dataframe, read_txt_as_list, get_all_files_with_extension_from_folder
from utilities import FACES_DIRECTORY, ROOT_DIR, SAMPLE_VIDEO_DIRECTORY

from torchvision.transforms import Normalize

from DeepfakeDataset import DeepfakeDataset

IMAGE_SIZE = 224
BATCH_SIZE = 64

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

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

# print(fake_to_original)
# print(fake_dataframe)

extracted_faces_names = get_all_files_with_extension_from_folder(FACES_DIRECTORY, ".png")
fake_videos = [x[:-4] for x in fake_videos]

result_fake = []
for video in fake_videos:
    result_fake.append([i for i in extracted_faces_names if i.startswith(video)])

flat_result = [item for sublist in result_fake for item in sublist]

flat_real = set(extracted_faces_names) - set(flat_result)

real_dataframe = list_to_dataframe(flat_real, "image_name")
fake_dataframe = list_to_dataframe(flat_result, "image_name")
real_dataframe["label"] = 0
fake_dataframe["label"] = 1

print(real_dataframe)
print(fake_dataframe)

# normalize_transform = Normalize(MEAN, STD)
unnormalize_transform = Unnormalize(MEAN, STD)


dataset = DeepfakeDataset(FACES_DIRECTORY, real_dataframe, fake_dataframe, sample_size=400)
plt.imshow(unnormalize_transform(dataset[0][0]).permute(1, 2, 0))
plt.show()

