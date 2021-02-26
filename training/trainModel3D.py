import logging
import config_logger
from functools import partial
import ntpath
import os
import random
import pickle
import time


from albumentations.pytorch.functional import img_to_tensor
from albumentations.augmentations.functional import resize
import cv2
import numpy as np
import pandas as pd
from timm.models.efficientnet_blocks import InvertedResidual
from timm.models.efficientnet import tf_efficientnet_l2_ns_475, tf_efficientnet_b0_ns, tf_efficientnet_b4_ns
import torch
from torch import distributions
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.nn import functional as F, Dropout, Linear, AdaptiveAvgPool2d
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


from utilities import SEQUENCE_DATAFRAME_PATH, REAL_FOLDER_TO_IDENTIFIERS_PATH, FAKE_FOLDER_TO_IDENTIFIERS_PATH, \
    SEQUENCE_DATAFRAME_TESTING_PATH, TESTING_FOLDER_TO_IDENTIFIERS_PATH, MODELS_DIECTORY
from training.augmentations import augmentation_pipeline_3D, gaussian_noise_transform_3D, \
    validation_augmentation_pipeline, put_to_center
from training.trainModel import collate_fn
from training.trainUtilities import MEAN, STD

log = logging.getLogger(__name__)


SEQUENCE_LENGTH = 5
MAX_ITERATIONS = 2500000
MAX_ITERATIONS_BATCH = 20000

model_save_path = os.path.join(MODELS_DIECTORY, "3Dnew_model")
pretrained_weights_path = os.path.join(MODELS_DIECTORY, "3Dmodel1.pth")

encoder_params_3D = {
    "tf_efficientnet_l2_ns_475": {
        "features": 5504,
        "init_op": partial(tf_efficientnet_l2_ns_475,
                           num_classes=1,
                           pretrained=True,
                           drop_rate=0.5)
    },
    "tf_efficientnet_b0_ns": {
        "features": 1280,
        "init_op": partial(tf_efficientnet_b0_ns,
                           num_classes=1,
                           pretrained=True)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns,
                           num_classes=1,
                           pretrained=True)
    }
}


class TestingDataset(Dataset):
    """ Deepfake Validation Dataset """

    def __init__(self, testing_dataframe, augmentations, path_to_identifiers_dictionary,
                 image_width=192, image_height=224):

        self.image_width = image_width
        self.image_height = image_height
        self.path_to_identifiers = path_to_identifiers_dictionary

        if 'index' in testing_dataframe:
            del testing_dataframe['index']

        self.testing_dataframe = testing_dataframe
        self.augmentate = augmentations

    def __getitem__(self, index):
        row = self.testing_dataframe.iloc[index]
        folder_name = row["faces_folder"]
        label = row["label"]

        identifiers = self.path_to_identifiers[folder_name]
        if len(identifiers) > 10:
            sequence = self.load_sequence(identifiers[4:9], folder_name)
        elif len(identifiers) >= 5:
            sequence = self.load_sequence(identifiers[0:5], folder_name)
        else:
            log.info(f"Failed to get Sequence: {folder_name}")
            return None

        return sequence, label

    def __len__(self):
        return len(self.testing_dataframe)

    def load_sequence(self, indices, image_folder):
        prefix = ntpath.basename(image_folder)
        full_prefix = os.path.join(image_folder, prefix)
        sequence_images = [self.load_img(full_prefix, identifier) for identifier in indices]
        transformed_image_sequence = []
        for img in sequence_images:
            image_transformed = self.augmentate(image=img)["image"]
            image_transformed = put_to_center(image_transformed, self.image_height)
            image_transformed = img_to_tensor(image_transformed, {"mean": MEAN,
                                                                  "std": STD})
            transformed_image_sequence.append(image_transformed)

        return np.stack(transformed_image_sequence)

    def load_img(self, sequence_path, idx):
        full_image_path = sequence_path + "_{}.png".format(idx)
        img = cv2.imread(full_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class DeepFakeDataset3D(Dataset):
    """Deepfake dataset"""

    def __init__(self, sequence_dataframe, real_dictionary_to_identifiers,
                 fake_dictionary_to_identifiers, augmentations, image_width=192, image_height=224):

        self.image_width = image_width
        self.image_height = image_height
        if 'index' in sequence_dataframe:
            del sequence_dataframe['index']

        self.augmentate = augmentations
        self.df = sequence_dataframe
        self.real_to_identifiers = real_dictionary_to_identifiers
        self.fake_to_identifiers = fake_dictionary_to_identifiers

    def __getitem__(self, index):
        row = self.df.iloc[index]
        real_image_folder = row["real_image_folder"]
        fake_image_folder = row["fake_image_folder"]

        real_identifiers = self.real_to_identifiers[real_image_folder]
        prev_state = random.getstate()
        if len(real_identifiers) > 10:
            real_sequence = self.load_sequence(real_identifiers[4:9], real_image_folder, prev_state)
        elif len(real_identifiers) >= 5:
            real_sequence = self.load_sequence(real_identifiers[0:5], real_image_folder, prev_state)
        else:
            log.info(f"Failed to get Real Sequence: {real_image_folder}")
            return None

        random.setstate(prev_state)
        fake_identifiers = self.fake_to_identifiers[fake_image_folder]
        if len(fake_identifiers) > 10:
            fake_sequence = self.load_sequence(fake_identifiers[4:9], fake_image_folder, prev_state)
        elif len(fake_identifiers) >= 5:
            fake_sequence = self.load_sequence(fake_identifiers[0:5], fake_image_folder, prev_state)
        else:
            log.info(f"Failed to get Fake Sequence: {fake_image_folder}")
            return None

        sequence_pair = {
            "real": real_sequence,
            "fake": fake_sequence
        }

        return sequence_pair

    def __len__(self):
        return len(self.df)

    def load_sequence(self, indices, image_folder, prev_state):
        prefix = ntpath.basename(image_folder)
        full_prefix = os.path.join(image_folder, prefix)
        sequence_images = [self.load_img(full_prefix, identifier) for identifier in indices]
        transformed_image_sequence = []
        for img in sequence_images:
            random.setstate(prev_state)
            image_transformed = self.augmentate(image=img)["image"]
            image_transformed = resize(image_transformed, height=self.image_height,
                                       width=self.image_width)
            image_transformed = gaussian_noise_transform_3D(image=image_transformed)["image"]
            image_transformed = img_to_tensor(image_transformed, {"mean": MEAN,
                                                                  "std": STD})
            transformed_image_sequence.append(image_transformed)

        return np.stack(transformed_image_sequence)

    def load_img(self, sequence_path, idx):
        full_image_path = sequence_path + "_{}.png".format(idx)
        img = cv2.imread(full_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class DeepfakeClassifier3D(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.encoder = encoder_params_3D["tf_efficientnet_l2_ns_475"]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params_3D["tf_efficientnet_l2_ns_475"]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DeepfakeClassifier3D_V2(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.encoder = encoder_params_3D["tf_efficientnet_b0_ns"]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params_3D["tf_efficientnet_b0_ns"]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class DeepfakeClassifier3D_V3(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.encoder = encoder_params_3D["tf_efficientnet_b4_ns"]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params_3D["tf_efficientnet_b4_ns"]["features"], 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ConvolutionExpander(nn.Module):
    def __init__(self, in_channels, out_channels, length):
        super(ConvolutionExpander, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.length = length

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        x = x.view(batch_size // self.length, self.length, in_channels, height, width)
        x = self.conv(x.transpose(1, 2).contiguous()).transpose(2, 1).contiguous()
        x = x.flatten(0, 1)
        return x


def evaluate(model, minimum_loss):
    model.eval()
    running_loss = 0
    total_examples_test = 0
    debugging_loss = 0

    for sequence, labels in testing_loader:
        with torch.no_grad():
            sequence = sequence.to(gpu)
            labels = labels.to(gpu)

            sequence = sequence.squeeze()
            outputs = model(sequence.flatten(0, 1))
            y_pred = outputs.squeeze()

            labels = labels.type_as(y_pred)

            loss = criterion(y_pred, labels.repeat_interleave(SEQUENCE_LENGTH))

            # need to track all images
            total_examples_test += sequence.size(0)
            running_loss += loss.item() * sequence.size(0)
            debugging_loss += loss.item()

    log.info('Debugging Loss: {:4f}'.format(debugging_loss))
    total_loss = running_loss / total_examples_test
    log.info('Validation Loss: {:4f}'.format(total_loss))

    if total_loss < minimum_loss:
        minimum_loss = total_loss

    return minimum_loss


def train_model(model, criterion, optimizer, scheduler, epochs):
    since = time.time()
    minimum_loss = 0.45
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
            # Will need to debug the data loader
            iteration += 1
            batch_number += 1

            fake_image_sequence = pairs['fake'].to(gpu)
            real_image_sequence = pairs['real'].to(gpu)

            target = probability_distribution.sample((len(real_image_sequence),)).float().to(gpu)
            fake_weight = target.view(-1, 1, 1, 1, 1)

            input_tensor = (1.0 - fake_weight) * real_image_sequence + fake_weight * fake_image_sequence

            current_batch_size = input_tensor.shape[0]
            total_examples += current_batch_size

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(input_tensor.flatten(0, 1))
            y_pred = outputs.squeeze()
            target = target.repeat_interleave(SEQUENCE_LENGTH)

            loss = criterion(y_pred, target)

            loss.backward()
            optimizer.step()

            # statistics
            running_training_loss += loss.item() * current_batch_size
            if batch_number % 250 == 0 and batch_number != 0:
                log.info("New 250 batches are evaluated")
                log.info("Batch Number: {}".format(batch_number))
                time_elapsed = time.time() - since
                log.info('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                max_lr = max(param_group["lr"] for param_group in optimizer.param_groups)
                log.info("iteration: {}, max_lr: {}".format(iteration, max_lr))

            if batch_number >= MAX_ITERATIONS_BATCH:
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
            path_model = model_save_path + str(epoch) + ".pth"
            torch.save(model.state_dict(), path_model)

    time_elapsed = time.time() - since

    log.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    log.info('Loss: {:4f}'.format(minimum_loss))

    return model


if __name__ == '__main__':
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("Program Started")
    log.info(f"GPU value: {gpu}")

    sequence_dataframe = pd.read_csv(SEQUENCE_DATAFRAME_PATH)
    with open(REAL_FOLDER_TO_IDENTIFIERS_PATH, 'rb') as handle:
        real_folder_to_identifiers = pickle.load(handle)
    with open(FAKE_FOLDER_TO_IDENTIFIERS_PATH, 'rb') as handle:
        fake_folder_to_identifiers = pickle.load(handle)

    batch_size = 4
    batch_size_testing = 4
    num_workers = 2

    train_dataset = DeepFakeDataset3D(sequence_dataframe, real_folder_to_identifiers, fake_folder_to_identifiers,
                                      augmentation_pipeline_3D())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn, drop_last=True)

    # need to create testing dataset
    testing_sequence_dataframe = pd.read_csv(SEQUENCE_DATAFRAME_TESTING_PATH)
    with open(TESTING_FOLDER_TO_IDENTIFIERS_PATH, 'rb') as handle:
        testing_folder_to_identifiers = pickle.load(handle)

    testing_dataset = TestingDataset(testing_sequence_dataframe, validation_augmentation_pipeline(),
                                     testing_folder_to_identifiers)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size_testing, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

    log.info(f"Dataloaders Created")

    model = DeepfakeClassifier3D_V3()

    for module in model.modules():
        if isinstance(module, InvertedResidual):
            if module.exp_ratio != 1.0:
                expansion_con = module.conv_pw
                expander = ConvolutionExpander(expansion_con.in_channels, expansion_con.out_channels, SEQUENCE_LENGTH)
                # 5 dimension tensor and we take third dimension
                expander.conv.weight.data[:, :, 0, :, :].copy_(expansion_con.weight.data / 3)
                expander.conv.weight.data[:, :, 1, :, :].copy_(expansion_con.weight.data / 3)
                expander.conv.weight.data[:, :, 2, :, :].copy_(expansion_con.weight.data / 3)
                module.conv_pw = expander

    model.load_state_dict(torch.load(pretrained_weights_path))
    model.to(gpu)

    log.info("Model is initialised")

    criterion = F.binary_cross_entropy_with_logits
    optimizer_ft = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,
                             weight_decay=1e-4, nesterov=True)
    lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.9)

    probability_distribution = distributions.beta.Beta(0.5, 0.5)

    log.info("Training Begins")

    history = {
        "train": [],
        "val": []
    }

    model = train_model(model, criterion, optimizer_ft, lr_scheduler, 25)

