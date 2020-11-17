import pandas as pd

from torch.utils.data import Dataset

from training.trainUtilities import load_image

from utilities import FACES_DIRECTORY


class DeepfakeDataset(Dataset):
    """Deepfake dataset"""

    def __init__(self, images_dir, frames_dataframe,
                 sample_size=20, image_size=224, seed=1):
        self.images_dir = images_dir
        self.image_size = image_size

        if sample_size:
            real_videos_dataframe = real_videos_dataframe.sample(sample_size, random_state=seed)
            fake_videos_dataframe = fake_videos_dataframe.sample(sample_size, random_state=seed)
            self.df = pd.concat([real_videos_dataframe, fake_videos_dataframe])

        else:
            self.df = pd.concat([real_videos_dataframe, fake_videos_dataframe])

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = row["image_name"]
        label = row["label"]
        img = load_image(image_name, self.images_dir, self.image_size)

        return img, label

    def __len__(self):
        len(self.df)