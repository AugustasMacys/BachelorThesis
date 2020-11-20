import pandas as pd
import os

from torch.utils.data import Dataset

from training.trainUtilities import load_image

from utilities import ROOT_DIR

FAKE_FACES_DIRECTORY = os.path.join(ROOT_DIR, "extracted_fake_images")
REAL_FACES_DIRECTORY = os.path.join(ROOT_DIR, "extracted_real_images")


class DeepfakeDataset(Dataset):
    """Deepfake dataset"""

    def __init__(self, frames_dataframe,
                 sample_size=None, image_size=224):

        self.image_size = image_size
        if 'index' in frames_dataframe:
            del frames_dataframe['index']

        mixed_dataframe = pd.DataFrame(frames_dataframe.stack().reset_index(drop=True),
                                       columns=["picture_name"])

        if sample_size:
            self.df = mixed_dataframe[:sample_size*2] # multiply by 2 because of real and fakes
        else:
            self.df = mixed_dataframe

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = row["picture_name"] + ".png"

        if index % 2 == 0:
            label = 1
            image_dir = FAKE_FACES_DIRECTORY
        else:
            image_dir = REAL_FACES_DIRECTORY
            label = 0

        img = load_image(image_name, image_dir, self.image_size)

        return img, label

    def __len__(self):
        return len(self.df)
