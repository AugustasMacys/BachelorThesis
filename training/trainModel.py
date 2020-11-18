import os
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch

from training.trainUtilities import Unnormalize
from utilities import DATAFRAMES_DIRECTORY, NOISY_STUDENT_DIRECTORY

from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch.nn as nn

from efficientnet_pytorch import EfficientNet

from DeepfakeDataset import DeepfakeDataset

IMAGE_SIZE = 224
BATCH_SIZE = 64

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

NOISY_STUDENT_WEIGHTS_FILENAME = os.path.join(NOISY_STUDENT_DIRECTORY, "noisy-student-efficientnet-b0.pth")

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


def evaluate(net, data_loader, device, silent=False):
    net.train(False)

    binary_cross_entropy_loss = 0
    total_examples = 0

    with tqdm.tqdm(total = len(data_loader), desc="Evaluation Function", leave=False,
              disable=silent) as bar:
        for batch_index, data in enumerate(data_loader):
            with torch.no_grad():
                batch_size = data[0].shape[0]
                x = data[0].to(device) # should be for GPU
                y_true = data[1].to(device).float()
                y_pred = net(x)
                y_pred = y_pred.squeeze()

                binary_cross_entropy_loss += F.binary_cross_entropy_with_logits(
                    y_pred, y_true).item() * batch_size

            total_examples += batch_size
            bar.update()

            if silent:
                return binary_cross_entropy_loss

            else:
                print("Binary Cross Entropy Loss: %.4f" %(binary_cross_entropy_loss))


def fit(epochs):

    global history, iteration, epochs_done, learning_rate

    with tqdm(total=len(train_loader), leave=False) as bar:
        for epoch in range(epochs):
            bar.reset()
            bar.set_description("Epoch %d" % (epochs_done + 1))

            binary_cross_entropy_loss = 0
            total_examples = 0

            net.train(True)

            for batch_idx, data in enumerate(train_loader):
                batch_size = data[0].shape[0]
                x = data[0].to(gpu)
                y_true = data[1].to(gpu).float()

                optimizer.zero_grad()

                y_pred = net(x)
                y_pred = y_pred.squeeze()

                loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
                loss.backward()
                optimizer.step()

                average_batch_binary_cross_entropy = loss.item()
                binary_cross_entropy_loss += batch_size * average_batch_binary_cross_entropy

                history["train_binary_cross_entropy"].append(batch_bce)

                total_examples += batch_size
                iteration += 1
                bar.update()

            binary_cross_entropy_loss /= total_examples
            epochs_done += 1

            print("Epoch: %d, train loss: %.4f" % (epochs_done, binary_cross_entropy_loss))

            validation_loss = evaluate(net, val_loader, device=gpu, silent=True)

            history["validation_binary_cross_entropy"].append(validation_loss)
            print("           val BCE: %.4f" % (validation_loss))
            print("")


class Effnet(nn.Module):

    def __init__(self):
        super(Effnet, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0', weights_path=NOISY_STUDENT_WEIGHTS_FILENAME)
        self.model.fc = nn.Linear(1280, 1)


if __name__ == '__main__':
    train_loader, validation_loader = create_data_loaders(frames_dataframe, BATCH_SIZE, 2)
    X, y = next(iter(validation_loader))
    plt.imshow(unnormalize_transform(X[0]).permute(1, 2, 0))
    plt.show()
    print(y[0])
