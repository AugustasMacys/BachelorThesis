from torch.utils.data import Dataset

from training.trainUtilities import load_image


class DeepfakeDataset(Dataset):
    """Deepfake dataset"""

    def __init__(self, frames_dataframe, augmentations, transformations,
                 image_size=224):

        self.image_size = image_size
        if 'index' in frames_dataframe:
            del frames_dataframe['index']

        self.augmentate = augmentations
        self.transform = transformations

        self.df = frames_dataframe

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_name = row["image_path"]
        label = row["label"]
        img = load_image(image_name)
        img = self.augmentate(image=img)["image"]

        return self.transform(img), label

    def __len__(self):
        return len(self.df)
