from PIL import Image
import logging
import torch
import os


import numpy as np


from torch.utils.data import Dataset
import albumentations.augmentations.functional as albumentations_F
from torchvision.transforms import functional as F


from utilities import MASK_REAL_PATH


log = logging.getLogger(__name__)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


tensorize = ToTensor()


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


normalise = Normalize(mean=MEAN, std=STD)


class XRayDataset(Dataset):
    """X-ray segmentation dataset"""

    def __init__(self, real_frames_dataframe, fake_frames_dataframe,
                 augmentations, mask_folder, image_size=224):

        self.mask_folder = mask_folder
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

        # Always will exist
        mask_real = Image.open(MASK_REAL_PATH).convert("L")

        folder_mask = real_image_name.split("\\")[3]
        identifier_mask = real_image_name.split("\\")[4]
        path_fake_mask = os.path.join(self.mask_folder, folder_mask, identifier_mask)

        # Might not exist
        try:
            mask_fake = Image.open(path_fake_mask).convert("L")
        except FileNotFoundError:
            log.info("Fake Mask not found: {}".format(path_fake_mask))
            return None

        img_real = np.array(img_real)
        mask_real = np.array(mask_real)

        img_fake = np.array(img_fake)
        mask_fake = np.array(mask_fake)
        transformed_real_images = self.augmentate(image=img_real, image2=mask_real)
        img_real = transformed_real_images["image"]
        mask_real = transformed_real_images["image2"]
        # due to isotropic resize need to resize mask
        mask_real = albumentations_F.resize(mask_real, height=56, width=56)

        transformed_fake_images = self.augmentate(image=img_fake, image2=mask_fake)
        img_fake = transformed_fake_images["image"]
        mask_fake = transformed_fake_images["image2"]
        # due to isotropic resize need to resize mask
        mask_fake = albumentations_F.resize(mask_fake, height=56, width=56)
        # values between 0 and 1
        mask_fake = (mask_fake - mask_fake.min()) / (np.ptp(mask_fake))

        img_real, mask_real = tensorize(image=img_real, target=mask_real)
        img_fake, mask_fake = tensorize(image=img_fake, target=mask_fake)

        img_real, mask_real = normalise(image=img_real, target=mask_real)
        img_fake, mask_fake = normalise(image=img_fake, target=mask_fake)

        images_with_masks = {
            "real_mask": mask_real,
            "fake_mask": mask_fake,
            "real_image": img_real,
            "fake_image": img_fake
        }

        return images_with_masks

    def __len__(self):
        return len(self.real_df)
