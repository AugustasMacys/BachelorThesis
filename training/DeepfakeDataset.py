import numpy as np
from PIL import Image

from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset

from training.trainUtilities import MEAN, STD


class DeepfakeDataset(Dataset):
    """Deepfake dataset"""

    def __init__(self, frames_dataframe, augmentations,
                 image_size=224):

        self.image_size = image_size
        if 'index' in frames_dataframe:
            del frames_dataframe['index']

        self.augmentate = augmentations

        self.df = frames_dataframe

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = row["image_path"]
        label = row["label"]
        img = Image.open(image_name).convert("RGB")
        img = np.array(img)
        img = self.augmentate(image=img)["image"]
        img = img_to_tensor(img, {"mean": MEAN,
                                  "std": STD})

        return img, label

    def __len__(self):
        return len(self.df)
