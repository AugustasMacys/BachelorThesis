from glob import glob
import os.path
import numpy as np
from PIL import Image
import re

from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset

from training.trainUtilities import MEAN, STD


class DeepfakeDataset(Dataset):
    """Deepfake dataset"""

    def __init__(self, real_frames_dataframe, fake_frames_dataframe,
                 augmentations, image_size=224):

        self.image_size = image_size
        if 'index' in real_frames_dataframe:
            del real_frames_dataframe['index']

        if 'index' in fake_frames_dataframe:
            del fake_frames_dataframe['index']

        self.augmentate = augmentations

        self.real_df = real_frames_dataframe
        self.fake_df = fake_frames_dataframe

    def __getitem__(self, index):
        row_real = self.real_df.iloc[index]
        row_fake = self.fake_df.iloc[index]

        real_image_name = row_real["image_path"]
        fake_image_name = row_fake["image_path"]
        current_identifier = -1
        try:
            img_real = Image.open(real_image_name).convert("RGB")
        except FileNotFoundError:
            img_real, current_identifier = self.find_new_identifier_and_image(real_image_name)

        try:
            if current_identifier != -1:
                new_current_identifier = int(re.findall(r'\d+', fake_image_name)[0])
                fake_image_name = fake_image_name.replace(str(current_identifier), str(new_current_identifier))
                current_identifier = -1

            img_fake = Image.open(fake_image_name).convert("RGB")
        except FileNotFoundError:
            img_fake, _ = self.find_new_identifier_and_image(fake_image_name)


        img = np.array(img)
        img = self.augmentate(image=img)["image"]
        img = img_to_tensor(img, {"mean": MEAN,
                                  "std": STD})

        return img, label

    def __len__(self):
        return len(self.df)

    def smallest_number_than(self, number, num_list):
        return min(num_list, key=lambda x: (abs(x - number), x))

    def biggest_number_than(self, number, num_list):
        return min(filter(lambda x: x > number, num_list))

    def find_new_identifier_and_image(self, image_name):
        current_identifier = int(re.findall(r'\d+', image_name)[0])
        directory = os.path.dirname(image_name)
        glob_string = os.path.join(directory, '*')
        list_files = glob(glob_string)
        identifiers = [re.findall(r'\d+', x) for x in list_files]
        smaller_identifier = self.smallest_number_than(current_identifier, identifiers)
        if smaller_identifier == current_identifier:
            bigger_identifier = self.biggest_number_than(current_identifier, identifiers)
            new_current_identifier = bigger_identifier
        else:
            new_current_identifier = smaller_identifier

        image_name = image_name.replace(str(current_identifier), str(new_current_identifier))
        img = Image.open(image_name).convert("RGB")
        current_identifier = new_current_identifier
        return img, current_identifier

