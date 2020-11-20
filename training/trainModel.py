import os
import pandas as pd

import torch
import time
import copy

from training.trainUtilities import Unnormalize
from utilities import DATAFRAMES_DIRECTORY, NOISY_STUDENT_DIRECTORY, MODELS_DIECTORY

from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim

from efficientnet_pytorch import EfficientNet

from DeepfakeDataset import DeepfakeDataset

IMAGE_SIZE = 224
BATCH_SIZE = 16

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

NOISY_STUDENT_WEIGHTS_FILENAME = os.path.join(NOISY_STUDENT_DIRECTORY, "noisy-student-efficientnet-b0.pth")

unnormalize_transform = Unnormalize(MEAN, STD)


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


def train_model(model, criterion, optimizer, scheduler, epochs):

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    mimimum_loss = 10000000

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(gpu)
                labels = labels.to(gpu)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    y_pred = outputs.squeeze()
                    #
                    # labels = labels.unsqueeze(1)
                    # labels = labels.float()
                    labels = labels.type_as(y_pred)

                    loss = criterion(y_pred, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and mimimum_loss > epoch_loss:
                mimimum_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Loss: {:4f}'.format(mimimum_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    frames_dataframe = pd.read_csv(os.path.join(DATAFRAMES_DIRECTORY, "frames_dataframe.csv"))
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader = create_data_loaders(frames_dataframe, BATCH_SIZE, 2)
    dataloaders = {
        "train": train_loader,
        "val": validation_loader
    }

    dataset_size = {
        "train": len(train_loader.dataset),
        "val": len(validation_loader.dataset)
    }

    model = EfficientNet.from_pretrained('efficientnet-b0', weights_path=NOISY_STUDENT_WEIGHTS_FILENAME,
                                         num_classes=1)

    # freeze parameters so that gradients are not computed
    for name, param in model.named_parameters():
        param.requires_grad = False

    model._fc = nn.Linear(1280, 1)

    model = model.to(gpu)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    criterion = F.binary_cross_entropy_with_logits

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_ft = optim.SGD(model._fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, 25)
    model_save_path = MODELS_DIECTORY
    torch.save(model.state_dict(), model_save_path)


    # print([k for k, v in net.named_parameters() if v.requires_grad])
    # for name, params in net.named_parameters():
    #     print(name)

    # out = net(torch.zeros((10, 3, IMAGE_SIZE, IMAGE_SIZE)).to(gpu))
    # X, y = next(iter(validation_loader))
    # plt.imshow(unnormalize_transform(X[0]).permute(1, 2, 0))
    # plt.show()
    # print(y[0])


