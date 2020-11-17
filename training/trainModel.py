import os
import pandas as pd

import matplotlib.pyplot as plt

from training.trainUtilities import Unnormalize
from utilities import DATAFRAMES_DIRECTORY

from torch.utils.data import DataLoader

from DeepfakeDataset import DeepfakeDataset

IMAGE_SIZE = 224
BATCH_SIZE = 64

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

unnormalize_transform = Unnormalize(MEAN, STD)

frames_dataframe = pd.read_csv(os.path.join(DATAFRAMES_DIRECTORY, "frames_dataframe.csv"), )
print(frames_dataframe)


def train_validation_split(metadata_dataframe, frac=0.2):

    n = int(len(metadata_dataframe) * frac)
    validation_dataframe = metadata_dataframe.iloc[0:n]
    train_dataframe = metadata_dataframe.iloc[n:].reset_index()

    return train_dataframe, validation_dataframe


def create_data_loaders(frames_dataframe, batch_size, num_workers):
    train_df, val_df = train_validation_split(frames_dataframe)

    train_dataset = DeepfakeDataset(train_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    validation_dataset = DeepfakeDataset(val_df)

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True)

    return train_loader, validation_loader

# def evaluate(net, data_loader, device, silent=False):


if __name__ == '__main__':
    train_loader, validation_loader = create_data_loaders(frames_dataframe, BATCH_SIZE, 2)
    X, y = next(iter(validation_loader))
    plt.imshow(unnormalize_transform(X[0]).permute(1, 2, 0))
    plt.show()
    print(y[0])
