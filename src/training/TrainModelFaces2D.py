import logging
from src import ConfigLogger
import os
from functools import partial
import pandas as pd
import time

from src.training.Augmentations import augmentation_pipeline, validation_augmentation_pipeline, \
    xray_augmentation_pipeline, augmentation_preprocessing_pipeline
from src.Utilities import MODELS_DIRECTORY, VALIDATION_DATAFRAME_PATH, \
    PAIR_REAL_DATAFRAME, PAIR_FAKE_DATAFRAME, MASKS_FOLDER,\
    PREVIEW_TEST, PREVIEW_TRAIN, FACE_XRAY_FULL_DATAFRAME
from src.training.DeepfakeDataset import DeepfakeDataset, ValidationDataset
from src.training_xray.Dataset_XRay import XRayDataset

from timm.models.efficientnet import tf_efficientnet_b4_ns
import torch
from torch import distributions
from torch.utils.data import DataLoader
from torch.nn import functional as F, AdaptiveAvgPool2d, Dropout, Linear
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

log = logging.getLogger(__name__)


MAX_ITERATIONS = 120000
BATCHES_PER_EPOCH = 6000

model_save_path = os.path.join(MODELS_DIRECTORY, "2DModel.pth")


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


encoder_params = {
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns,
                           num_classes=1,
                           pretrained=True,
                           drop_path_rate=0.2)
    }
}


class MyResNeXt(models.resnet.ResNet):
    def __init__(self):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3],
                                        groups=32,
                                        width_per_group=4)

        # Override the existing FC layer with a new one.
        self.fc = nn.Linear(2048, 1)


class DeepfakeClassifier(nn.Module):
    def __init__(self, dropout_rate=0.2):
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


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def create_data_loaders(batch_size, num_workers, non_existing_files=None,
                        X_RAY=False, preview=False):
    if X_RAY:
        if preview:
            train_df = pd.read_csv(PREVIEW_TRAIN)
        else:
            train_df = pd.read_csv(FACE_XRAY_FULL_DATAFRAME)

        train_dataset = XRayDataset(train_df, xray_augmentation_pipeline(),
                                    MASKS_FOLDER)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True, collate_fn=collate_fn, drop_last=False)
    else:
        train_real_df = pd.read_csv(PAIR_REAL_DATAFRAME)
        train_fake_df = pd.read_csv(PAIR_FAKE_DATAFRAME)

        train_dataset = DeepfakeDataset(train_real_df, train_fake_df, augmentation_pipeline(),
                                        augmentation_preprocessing_pipeline(),
                                        non_existing_files)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True, collate_fn=collate_fn)

    if preview:
        val_df = pd.read_csv(PREVIEW_TEST)
    else:
        val_df = pd.read_csv(VALIDATION_DATAFRAME_PATH)

    if X_RAY:
        validation_dataset = ValidationDataset(val_df, validation_augmentation_pipeline())
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True)
    else:
        validation_dataset = ValidationDataset(val_df, validation_augmentation_pipeline())
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True)

    return train_loader, validation_loader


def evaluate(model, minimum_loss):
    model.eval()
    running_loss = 0

    for images, labels in validation_loader:
        with torch.no_grad():
            images = images.to(gpu)
            labels = labels.to(gpu)

            outputs = model(images)
            y_pred = outputs.squeeze()
            labels = labels.type_as(y_pred)
            loss = criterion(y_pred, labels)

            running_loss += loss.item() * images.size(0)

    total_loss = running_loss / dataset_size["val"]
    log.info('Validation Loss: {:4f}'.format(total_loss))

    if total_loss < minimum_loss:
        minimum_loss = total_loss

    return minimum_loss


def train_model(model, criterion, optimizer, scheduler, epochs):
    since = time.time()
    minimum_loss = 0.69
    iteration = 0

    for epoch in range(epochs):
        batch_number = 0
        if iteration == MAX_ITERATIONS:
            break

        log.info('Epoch {}/{}'.format(epoch, epochs - 1))
        log.info('-' * 10)

        model.train()  # Set model to training mode

        running_training_loss = 0
        total_examples = 0
        for pairs in train_loader:
            iteration += 1
            batch_number += 1
            fake_images = pairs['fake'].to(gpu)
            real_images = pairs['real'].to(gpu)

            target = beta_probability_distribution.sample((len(fake_images),)).float().to(gpu)
            fake_weight = target.view(-1, 1, 1, 1)

            mixed_faces = (1.0 - fake_weight) * real_images + fake_weight * fake_images

            current_batch_size = mixed_faces.shape[0]
            total_examples += current_batch_size

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(mixed_faces)
            y_pred = outputs.squeeze()

            loss = criterion(y_pred, target)

            loss.backward()
            optimizer.step()

            # statistics
            running_training_loss += loss.item() * current_batch_size
            if batch_number % 250 == 0:
                log.info("New 250 batches are evaluated")
                log.info("Batch Number: {}".format(batch_number))
                time_elapsed = time.time() - since
                log.info('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                max_lr = max(param_group["lr"] for param_group in optimizer.param_groups)
                log.info("iteration: {}, max_lr: {}".format(iteration, max_lr))

            if batch_number >= BATCHES_PER_EPOCH:
                break

        epoch_loss = running_training_loss / total_examples
        log.info('Training Loss: {:.4f}'.format(epoch_loss))
        history["train"].append(epoch_loss)

        # Calculate Validation Loss
        validation_loss = evaluate(model, minimum_loss)
        history["val"].append(validation_loss)
        log.info(history)
        scheduler.step()

        # deep copy the model
        if validation_loss < minimum_loss:
            minimum_loss = validation_loss
            log.info("Minimum loss is: {}".format(minimum_loss))
            torch.save(model.state_dict(), model_save_path)

    time_elapsed = time.time() - since

    log.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    log.info('Loss: {:4f}'.format(minimum_loss))

    # load best model weights
    model.load_state_dict(torch.load(model_save_path))
    return model


if __name__ == '__main__':
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Program Started")
    log.info(f"GPU value: {gpu}")

    BATCH_SIZE = 28
    train_loader, validation_loader = create_data_loaders(BATCH_SIZE, 4)

    dataset_size = {
        "val": len(validation_loader.dataset)
    }
    val_size = dataset_size["val"]

    log.info(f"Dataloaders Created")
    log.info(f"Size of validation dataloader is {val_size}")

    model = DeepfakeClassifier()

    model = model.to(gpu)
    log.info("Model is initialised")

    # uncomment if you want to do transfer learning
    # freeze_until(model, "encoder.blocks.5.7.bn3.bias")

    criterion = F.binary_cross_entropy_with_logits
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                             weight_decay=1e-4, nesterov=True)
    lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.9)
    beta_probability_distribution = distributions.beta.Beta(0.5, 0.5)

    log.info("Training Begins")

    history = {
        "train": [],
        "val": []
    }

    model = train_model(model, criterion, optimizer_ft, lr_scheduler, 25)
