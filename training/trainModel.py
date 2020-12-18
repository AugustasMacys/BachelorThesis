import logging
from logging.handlers import RotatingFileHandler
import os
from functools import partial
import pandas as pd
import time

from training.augmentations import augmentation_pipeline, validation_augmentation_pipeline, transformation
from training.trainUtilities import Unnormalize
from utilities import NOISY_STUDENT_DIRECTORY, MODELS_DIECTORY, \
    VALIDATION_DATAFRAME_PATH, TRAINING_DATAFRAME_PATH, RESNET_FOLDER
from training.DeepfakeDataset import DeepfakeDataset

import torch
from timm.models.efficientnet import tf_efficientnet_b4_ns
from torch.utils.data import DataLoader
from torch.nn import functional as F, AdaptiveAvgPool2d, Dropout, Linear
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# from efficientnet_pytorch import EfficientNet

MAX_ITERATIONS = 80000
BATCHES_PER_EPOCH = 8000

RESNET_WEIGHTS = os.path.join(RESNET_FOLDER, "resnext50_32x4d-7cdf4587.pth")

encoder_params = {
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns,
                           num_classes=1,
                           pretrained=True,
                           drop_path_rate=0.2)
    }
}


class DeepfakeClassifier(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.encoder = encoder_params["tf_efficientnet_b4_ns"]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params["tf_efficientnet_b4_ns"]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        groups=32,
                                        width_per_group=4)

        self.load_state_dict(torch.load(RESNET_WEIGHTS))

        # Override the existing FC layer with a new one.
        self.fc = nn.Linear(2048, 1)


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


IMAGE_SIZE = 224
BATCH_SIZE = 64

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

NOISY_STUDENT_WEIGHTS_FILENAME = os.path.join(NOISY_STUDENT_DIRECTORY, "noisy-student-efficientnet-b0.pth")

unnormalize_transform = Unnormalize(MEAN, STD)

model_save_path = os.path.join(MODELS_DIECTORY, "lowest_loss_model.pth")


def create_data_loaders(batch_size, num_workers):
    train_df = pd.read_csv(TRAINING_DATAFRAME_PATH)
    val_df = pd.read_csv(VALIDATION_DATAFRAME_PATH)

    train_dataset = DeepfakeDataset(train_df, augmentation_pipeline())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    validation_dataset = DeepfakeDataset(val_df, validation_augmentation_pipeline())

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=True)

    return train_loader, validation_loader


def train_model(model, criterion, optimizer, scheduler, epochs):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    mimimum_loss = 10000000
    iteration = 0
    batch_number = 0
    for epoch in range(epochs):
        logging.info('Epoch {}/{}'.format(epoch, epochs - 1))
        logging.info('-' * 10)
        # print('Epoch {}/{}'.format(epoch, epochs - 1))
        # print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if phase == "train":
                    iteration += 1
                    batch_number += 1

                inputs = inputs.to(gpu)
                labels = labels.to(gpu)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    y_pred = outputs.squeeze()
                    labels = labels.type_as(y_pred)

                    loss = criterion(y_pred, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if batch_number % 500 == 0:
                    logging.info("New 500 batches are evaluated")
                    logging.info("Batch Number: {}".format(batch_number))
                    # print("Batch Number: {}".format(batch_number))
                    time_elapsed = time.time() - since
                    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
                        time_elapsed // 60, time_elapsed % 60))
                    # print('Training complete in {:.0f}m {:.0f}s'.format(
                    #     time_elapsed // 60, time_elapsed % 60))
                    max_lr = max(param_group["lr"] for param_group in optimizer.param_groups)
                    logging.info("iteration: {}, max_lr: {}".format(iteration, max_lr))

                # if batch_number >= BATCHES_PER_EPOCH:
                #     batch_number = 0
                #     break

                if phase == 'train':
                    scheduler.step()
                    # max_lr = max(param_group["lr"] for param_group in optimizer.param_groups)
                    # logging.info("iteration: {}, max_lr: {}".format(iteration, max_lr))
                    # print("iteration: {}, max_lr: {}".format(iteration, max_lr))

            if phase == "train":
                epoch_loss = running_loss / dataset_size[phase]
                # epoch_loss = running_loss / (BATCHES_PER_EPOCH * BATCH_SIZE) # Does not go through all size
            else:
                epoch_loss = running_loss / dataset_size[phase]

            logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and mimimum_loss > epoch_loss:
                mimimum_loss = epoch_loss
                logging.info("Minimum loss is: {}".format(mimimum_loss))
                # best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), model_save_path)

        # print()

    time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Loss: {:4f}'.format(mimimum_loss))

    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Loss: {:4f}'.format(mimimum_loss))

    # load best model weights
    model.load_state_dict(torch.load(model_save_path))
    return model


if __name__ == '__main__':

    handler = RotatingFileHandler(filename='../logs/training_log.log', maxBytes=20000000, backupCount=10)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(name)s, %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[handler])
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Program Started")
    logging.info(f"GPU value: {gpu}")

    train_loader, validation_loader = create_data_loaders(BATCH_SIZE, 6)

    dataloaders = {
        "train": train_loader,
        "val": validation_loader
    }

    dataset_size = {
        "train": len(train_loader.dataset),
        "val": len(validation_loader.dataset)
    }

    logging.info(f"Dataloaders Created, size of train_loader is: {len(train_loader.dataset)},"
                 f"val_loader is: {len(validation_loader.dataset)}")

    # model = DeepfakeClassifier()
    model = MyResNeXt()

    logging.info("Model is initialised")

    # model = EfficientNet.from_pretrained('efficientnet-b4', weights_path=NOISY_STUDENT_WEIGHTS_FILENAME,
    #                                      num_classes=1)
    # freeze parameters so that gradients are not computed
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    # model._fc = nn.Linear(1280, 1)

    model = model.to(gpu)
    freeze_until(model, "layer4.0.conv1.weight")
    criterion = F.binary_cross_entropy_with_logits
    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    lr_scheduler = lr_scheduler.LambdaLR(optimizer_ft, lambda iteration: (MAX_ITERATIONS - iteration) / MAX_ITERATIONS)

    logging.info("Training Begins")

    model = train_model(model, criterion, optimizer_ft, lr_scheduler, 25)
