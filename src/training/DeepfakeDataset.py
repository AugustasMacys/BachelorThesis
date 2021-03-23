import logging

# import cv2
import numpy as np
from PIL import Image

import albumentations.augmentations.functional as F
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset

from src.training.Augmentations import gaussian_noise_transform, put_to_center
from src.training.TrainUtilities import MEAN, STD

log = logging.getLogger(__name__)


class ValidationDataset(Dataset):
    """ Deepfake Validation Dataset """

    def __init__(self, validation_dataframe, augmentations,
                 image_size=224):

        self.image_size = image_size

        if 'index' in validation_dataframe:
            del validation_dataframe['index']

        self.validation_dataframe = validation_dataframe
        self.augmentate = augmentations

    def __getitem__(self, index):
        row = self.validation_dataframe.iloc[index]
        image_name = row["image_path"]
        label = row["label"]

        img = Image.open(image_name).convert("RGB")
        img = np.array(img)

        transformed_image = self.augmentate(image=img)["image"]
        transformed_image = put_to_center(transformed_image, self.image_size)
        transformed_image = img_to_tensor(transformed_image, {"mean": MEAN,
                                                              "std": STD})

        return transformed_image, label

    def __len__(self):
        return len(self.validation_dataframe)


class DeepfakeDataset(Dataset):
    """Deepfake dataset"""

    def __init__(self, real_frames_dataframe, fake_frames_dataframe,
                 augmentations, non_existing_files, image_size=224):

        # Should increase training speed as on second epoch will not need to catch exceptions
        self.non_existing_files = non_existing_files
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

        # Will go here from second epoch
        # if real_image_name in self.non_existing_files or fake_image_name in self.non_existing_files:
        #     return None

        try:
            img_real = Image.open(real_image_name).convert("RGB")
        except FileNotFoundError:
            log.info("Real Image not found: {}".format(real_image_name))
            return None
        try:
            img_fake = Image.open(fake_image_name).convert("RGB")
        except FileNotFoundError:
            log.info("Fake Image not found: {}".format(fake_image_name))
            return None

        img_real = np.array(img_real)
        img_fake = np.array(img_fake)
        transformed_images = self.augmentate(image=img_real, image2=img_fake)
        img_real = transformed_images["image"]
        img_fake = transformed_images["image2"]
        img_fake = F.resize(img_fake, height=224, width=224)
        gaussian_transformed_images = gaussian_noise_transform(image=img_real,
                                                               image2=img_fake)
        img_real = gaussian_transformed_images["image"]
        img_fake = gaussian_transformed_images["image2"]
        # print(img_real.shape)
        # print(img_fake.shape)
        # cv2.imwrite("test_real.png", img_real)
        # cv2.imwrite("test_fake.png", img_fake)
        img_real = img_to_tensor(img_real, {"mean": MEAN,
                                            "std": STD})
        img_fake = img_to_tensor(img_fake, {"mean": MEAN,
                                            "std": STD})

        pair = {
            "real": img_real,
            "fake": img_fake
        }

        return pair

    def __len__(self):
        return len(self.real_df)
