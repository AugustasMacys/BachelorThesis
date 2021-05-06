import logging


import albumentations.augmentations.functional as albumentations_F
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


from src.Utilities import MASK_REAL_PATH, MEAN, STD


log = logging.getLogger(__name__)



class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.float32)
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

    def __init__(self, dataframe, augmentations, mask_folder, image_size=224):
        self.mask_folder = mask_folder
        self.image_size = image_size

        if 'index' in dataframe:
            del dataframe['index']

        self.augmentate = augmentations
        self.df = dataframe

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = row["label"]

        image_name = row["image_path"]

        try:
            img = Image.open(image_name).convert("RGB")
            # (width, height)
            img = img.resize(size=(224, 224))
        except FileNotFoundError:
            log.info("Image not found: {}".format(image_name))
            return None

        # Always will exist
        if label == 0:
            mask = Image.open(MASK_REAL_PATH).convert("L")

        else:
            try:
                mask_path = row["mask_path"]
                mask = Image.open(mask_path).convert("L")
            except FileNotFoundError:
                log.info("Fake Mask not found: {}".format(mask_path))
                return None

        img = np.array(img)
        mask = np.array(mask)

        transformed_images = self.augmentate(image=img, image2=mask)
        img = transformed_images["image"]
        mask = transformed_images["image2"]
        # due to isotropic resize need to resize mask
        mask = albumentations_F.resize(mask, height=56, width=56,
                                       interpolation=cv2.INTER_NEAREST)

        # binarise 0 and 1
        if label == 1:
            mask = (mask - mask.min()) / (np.ptp(mask))

        img, mask = tensorize(image=img, target=mask)
        img, mask = normalise(image=img, target=mask)

        images_with_masks = {
            "img": img,
            "mask": mask,
            "label": label
        }

        return images_with_masks

    def __len__(self):
        return len(self.df)
