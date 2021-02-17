from functools import partial
import config_logger
import logging
import os
import random
import time

import ntpath
import numpy as np

import cv2

from glob import glob
from timm.models.efficientnet import tf_efficientnet_l2_ns_475
from torch.nn import functional as F, Dropout, Linear, AdaptiveAvgPool3d
from torch.optim import lr_scheduler
import torch.optim as optim
import torch
import torch.nn as nn

from albumentations.pytorch.functional import img_to_tensor

from timm.models.efficientnet_blocks import InvertedResidual
from torch.utils.data import Dataset

from torch import distributions

from training.augmentations import augmentation_pipeline_3D, gaussian_noise_transform_3D
from training.trainUtilities import MEAN, STD

log = logging.getLogger(__name__)


SEQUENCE_LENGTH = 5
MAX_ITERATIONS = 100000


encoder_params_3D = {
    "tf_efficientnet_l2_ns_475": {
        "features": 5504,
        "init_op": partial(tf_efficientnet_l2_ns_475,
                           num_classes=1,
                           pretrained=True,
                           drop_rate=0.5)
    }
}


class DeepFakeDataset3D(Dataset):
    """Deepfake dataset"""

    def __init__(self, sequence_dataframe, real_dictionary_to_identifiers,
                 fake_dictionary_to_identifiers, augmentations, image_width=224, image_height=192):

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
        if len(real_identifiers > 10):
            real_sequence = self.load_sequence(real_identifiers[4:9], real_image_folder, prev_state)
            # sequence = np.stack([self.load_img(full_prefix, identifier) for identifier in indices])
        elif len(real_identifiers) >= 5:
            real_sequence = self.load_sequence(real_identifiers[0:5], real_image_folder, prev_state)
            # sequence = np.stack([self.load_img(full_prefix, identifier) for identifier in indices])
        else:
            log.info("Failed to get Real Sequence")
            return None

        random.setstate(prev_state)
        fake_identifiers = self.fake_to_identifiers[fake_image_folder]
        if len(fake_identifiers > 10):
            fake_sequence = self.load_sequence(fake_identifiers[4:9], fake_image_folder, prev_state)
            # sequence = np.stack([self.load_img(full_prefix, identifier) for identifier in indices])
        elif len(fake_identifiers) >= 5:
            fake_sequence = self.load_sequence(fake_identifiers[0:5], fake_image_folder, prev_state)
            # sequence = np.stack([self.load_img(full_prefix, identifier) for identifier in indices])
        else:
            log.info("Failed to get Fake Sequence")
            return None

        sequence_pair = {
            "real": real_sequence,
            "fake": fake_sequence
        }

        return sequence_pair

    def load_sequence(self, indices, image_folder, prev_state):
        prefix = ntpath.basename(image_folder)
        full_prefix = os.path.join(image_folder, prefix)
        sequence_images = [self.load_img(full_prefix, identifier) for identifier in indices]
        transformed_image_sequence = []
        for img in sequence_images:
            random.setstate(prev_state)
            image_transformed = self.augmentate(image=img)["image"]
            image_transformed = gaussian_noise_transform_3D(image=image_transformed)["image"]
            image_transformed = img_to_tensor(image_transformed, {"mean": MEAN,
                                                                  "std": STD})
            transformed_image_sequence.append(image_transformed)

        return transformed_image_sequence

    def load_img(self, track_path, idx):
        full_image_path = track_path + "_{}.png".format(idx)
        img = cv2.imread(full_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


class DeepfakeClassifier3D(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.encoder = encoder_params_3D["tf_efficientnet_l2_ns_475"]["init_op"]()
        self.avg_pool = AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params_3D["tf_efficientnet_l2_ns_475"]["features"], 1)

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


def train_model(model, criterion, optimizer, scheduler, epochs):
    since = time.time()
    minimum_loss = 0.70
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
            target = probability_distribution.sample((len(fake_image_sequence),)).float().to(gpu)
            fake_weight = target.view(-1, 1, 1, 1, 1)

            input_tensor = (1.0 - fake_weight) * real_image_sequence + fake_weight * fake_image_sequence

            current_batch_size = input_tensor.shape[0]
            total_examples += current_batch_size

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(input_tensor)
            y_pred = outputs.squeeze()

            loss = criterion(y_pred, target)

            loss.backward()
            optimizer.step()

            # statistics
            running_training_loss += loss.item() * current_batch_size
            if batch_number % 500 == 0:
                log.info("New 500 batches are evaluated")
                log.info("Batch Number: {}".format(batch_number))
                time_elapsed = time.time() - since
                log.info('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                max_lr = max(param_group["lr"] for param_group in optimizer.param_groups)
                log.info("iteration: {}, max_lr: {}".format(iteration, max_lr))

            if batch_number >= BATCHES_PER_EPOCH:
                break

            # scheduler.step()

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
            # best_model_wts = copy.deepcopy(model.state_dict())
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

    ## create dataloaders

    log.info(f"Dataloaders Created")

    model = DeepfakeClassifier3D()

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

    model.to(gpu)

    log.info("Model is initialised")

    criterion = F.binary_cross_entropy_with_logits
    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                             weight_decay=1e-4, nesterov=True)
    lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.9)

    probability_distribution = distributions.beta.Beta(0.5, 0.5)

    log.info("Training Begins")

    history = {
        "train": [],
        "val": []
    }

    model = train_model(model, criterion, optimizer_ft, lr_scheduler, 25)


