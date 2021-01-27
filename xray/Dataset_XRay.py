from PIL import Image
import logging
import torch


import numpy as np


from torch.utils.data import Dataset
import albumentations.augmentations.functional as albumentations_F
from torchvision.transforms import functional as F


from utilities import MASK_REAL_PATH


log = logging.getLogger(__name__)


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class XRayDataset(Dataset):
    """X-ray segmentation dataset"""

    def __init__(self, real_frames_dataframe, fake_frames_dataframe,
                 augmentations, mask_folder, image_size=224):

        # # Should increase training speed as on second epoch will not need to catch exceptions
        # self.non_existing_files = non_existing_files
        self.mask_folder = mask_folder
        self.image_size = image_size
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

        # Might not exist
        try:
            mask_fake = Image.open(fake_image_name).convert("L")
        except FileNotFoundError:
            log.info("Fake Mask not found: {}".format(fake_image_name))
            return None

        mask_real = np.array(mask_real)
        mask_fake = np.array(mask_fake)
        # I decided to transform both masks the same way
        transformed_images = self.augmentate(image=mask_real, image2=mask_fake)
        mask_real = transformed_images["image"]
        mask_fake = transformed_images["image2"]
        mask_fake = albumentations_F.resize(mask_fake, height=224, width=224)

        # Just transform between 0 and 1 or smth

        images_with_masks = {
            "real": mask_real,
            "fake": mask_fake
        }

        return images_with_masks

    def __len__(self):
        return len(self.real_df)
