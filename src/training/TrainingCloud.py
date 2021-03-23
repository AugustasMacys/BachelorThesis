from functools import partial
import io
import logging
import ntpath
import os
import random
import pickle
from PIL import Image
import time


from albumentations import (
    HorizontalFlip, GaussianBlur, HueSaturationValue, DualTransform, GaussNoise, OneOf,
    Compose, RandomBrightnessContrast, ImageCompression, ShiftScaleRotate,
    PadIfNeeded, ToGray, FancyPCA, MotionBlur
)
from albumentations.pytorch.functional import img_to_tensor
from albumentations.augmentations.functional import resize
import cv2
from google.api_core.exceptions import ServiceUnavailable
import numpy as np
import pandas as pd
from timm.models.efficientnet_blocks import InvertedResidual
from timm.models.efficientnet import tf_efficientnet_l2_ns, tf_efficientnet_b8_ap, tf_efficientnet_b4_ns, \
    tf_efficientnet_b7_ns
import torch
from torch import distributions
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.nn import functional as F, Dropout, Linear, AdaptiveAvgPool2d
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEQUENCE_DATAFRAME_PATH = r"./dataframes/3dcnn_dataframe_cloud.csv"
REAL_FOLDER_TO_IDENTIFIERS_PATH = r"real_folder_to_identifiers.pickle"
FAKE_FOLDER_TO_IDENTIFIERS_PATH = r"fake_folder_to_identifiers.pickle"
SEQUENCE_DATAFRAME_TESTING_PATH = r"./dataframes/testing_dataframe_cloud.csv"
TESTING_FOLDER_TO_IDENTIFIERS_PATH = r"testing_folder_to_identifiers.pickle"
MODELS_DIECTORY = r"./trained_models"


NAME_FAKE = "fake_faces_training"
NAME_REAL = "real_faces_training"


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SEQUENCE_LENGTH = 5
MAX_ITERATIONS = 250000
MAX_ITERATIONS_BATCH = 20000

model_save_path = os.path.join(MODELS_DIECTORY, "3Dmodel")


# Helper Functions

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)


def augmentation_pipeline_3D(size_height=224, size_width=192):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussianBlur(blur_limit=3, p=0.05),
        MotionBlur(p=0.05),
        HorizontalFlip(),
        OneOf([
            IsotropicResize(max_side=size_height, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size_height, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size_height, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        PadIfNeeded(min_height=size_height, min_width=size_width, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5)]
    )


gaussian_noise_transform_3D = Compose([
        GaussNoise(p=0.1)]
    )

def validation_augmentation_pipeline(height=224, width=192):
    return Compose([
        IsotropicResize(max_side=height, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT),
    ])


def put_to_center(img, input_size):
    img = img[:input_size, :input_size]
    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    start_w = (input_size - img.shape[1]) // 2
    start_h = (input_size - img.shape[0]) // 2
    image[start_h:start_h + img.shape[0], start_w: start_w + img.shape[1], :] = img
    return image


encoder_params_3D = {
    "tf_efficientnet_l2_ns": {
        "features": 5504,
        "init_op": partial(tf_efficientnet_l2_ns,
                           num_classes=1,
                           pretrained=True,
                           drop_rate=0.5)
    },

    "tf_efficientnet_b8_ap": {
        "features": 2816,
        "init_op": partial(tf_efficientnet_b8_ap,
                           num_classes=1,
                           pretrained=True,
                           drop_rate=0.5)
    },
    "tf_efficientnet_b4_ns": {
        "features": 1792,
        "init_op": partial(tf_efficientnet_b4_ns,
                           num_classes=1,
                           pretrained=True)
    },

    "tf_efficientnet_b7_ns": {
        "features": 2560,
        "init_op": partial(tf_efficientnet_b7_ns,
                           num_classes=1,
                           pretrained=True,
                           drop_rate=0.5)
    }
}

# Dataset Classes


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
        # client = storage.Client()
        try:
            bucket = client.get_bucket(NAME_TEST)
        except:
            log.error("Cannot get bucket")
            return None
        row = self.testing_dataframe.iloc[index]
        folder_name = row["faces_folder"]
        label = row["label"]

        identifiers = self.path_to_identifiers[folder_name]
        if len(identifiers) > 10:
            sequence = self.load_sequence(identifiers[4:9], folder_name, bucket)
        elif len(identifiers) >= 5:
            sequence = self.load_sequence(identifiers[0:5], folder_name, bucket)
        else:
            log.info(f"Failed to get Sequence: {folder_name}")
            return None

        if sequence is None:
            log.info(f"Failed to get Sequence: {folder_name}")
            return None

        return sequence, label

    def __len__(self):
        return len(self.testing_dataframe)

    def load_sequence(self, indices, image_folder, bucket):
        prefix = ntpath.basename(image_folder)
        full_prefix = os.path.join(image_folder, prefix)
        sequence_images = [self.load_img(full_prefix, identifier, bucket) for identifier in indices]
        transformed_image_sequence = []
        for img in sequence_images:
            if img is None:
                return None

            image_transformed = self.augmentate(image=img)["image"]
            image_transformed = put_to_center(image_transformed, self.image_height)
            image_transformed = img_to_tensor(image_transformed, {"mean": MEAN,
                                                                  "std": STD})
            transformed_image_sequence.append(image_transformed)

        return np.stack(transformed_image_sequence)

    def load_img(self, sequence_path, idx, bucket):
        full_image_path = sequence_path + "_{}.png".format(idx)
        try:
            blob = bucket.get_blob(full_image_path)
            if blob is None:
                log.info(f"Cannot get Testing blob with path: {full_image_path}")
                return None
            blob = blob.download_as_string()
            blob_in_bytes = io.BytesIO(blob)
            img = np.asarray(Image.open(blob_in_bytes).convert("RGB"))
            return img
        except ServiceUnavailable:
            log.error("Service unavailable")
            return None


class DeepFakeDataset3D(Dataset):
    """Deepfake dataset"""

    def __init__(self, sequence_dataframe, real_dictionary_to_identifiers,
                 fake_dictionary_to_identifiers, augmentations,
                 # real_bucket, fake_bucket,
                 image_width=192, image_height=224):

        self.image_width = image_width
        self.image_height = image_height
        if 'index' in sequence_dataframe:
            del sequence_dataframe['index']

        self.augmentate = augmentations
        self.df = sequence_dataframe
        self.real_to_identifiers = real_dictionary_to_identifiers
        self.fake_to_identifiers = fake_dictionary_to_identifiers
        # client = storage.Client()
        # self.bucket_fake = client.get_bucket(NAME_FAKE)
        # self.bucket_real = client.get_bucket(NAME_REAL)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        real_image_folder = row["real_image_folder"]
        fake_image_folder = row["fake_image_folder"]

        # client = storage.Client()
        try:
            bucket_fake = client.get_bucket(NAME_FAKE)
            bucket_real = client.get_bucket(NAME_REAL)
        except:
            log.error("Cannot get bucket")
            return None

        real_identifiers = self.real_to_identifiers[real_image_folder]
        prev_state = random.getstate()
        if len(real_identifiers) > 10:
            real_sequence = self.load_sequence(real_identifiers[4:9], real_image_folder, prev_state,
                                               bucket_real)
        elif len(real_identifiers) >= 5:
            real_sequence = self.load_sequence(real_identifiers[0:5], real_image_folder, prev_state,
                                               bucket_real)
        else:
            log.info(f"Failed to get Real Sequence: {real_image_folder}")
            return None

        random.setstate(prev_state)
        fake_identifiers = self.fake_to_identifiers[fake_image_folder]
        if len(fake_identifiers) > 10:
            fake_sequence = self.load_sequence(fake_identifiers[4:9], fake_image_folder, prev_state,
                                               bucket_fake)
        elif len(fake_identifiers) >= 5:
            fake_sequence = self.load_sequence(fake_identifiers[0:5], fake_image_folder, prev_state,
                                               bucket_fake)
        else:
            log.info(f"Failed to get Fake Sequence: {fake_image_folder}")
            return None

        if fake_sequence is None:
            log.info(f"Failed to get Fake Sequence: {fake_image_folder}")
            return None

        if real_sequence is None:
            log.info(f"Failed to get Real Sequence: {real_image_folder}")
            return None

        sequence_pair = {
            "real": real_sequence,
            "fake": fake_sequence
        }

        return sequence_pair

    def __len__(self):
        return len(self.df)

    def load_sequence(self, indices, image_folder, prev_state, bucket):
        prefix = ntpath.basename(image_folder)
        full_prefix = os.path.join(image_folder, prefix)
        sequence_images = [self.load_img(full_prefix, identifier, bucket) for identifier in indices]
        transformed_image_sequence = []
        for img in sequence_images:
            if img is None:
                return None

            random.setstate(prev_state)
            image_transformed = self.augmentate(image=img)["image"]
            image_transformed = resize(image_transformed, height=self.image_height,
                                       width=self.image_width)
            image_transformed = gaussian_noise_transform_3D(image=image_transformed)["image"]
            image_transformed = img_to_tensor(image_transformed, {"mean": MEAN,
                                                                  "std": STD})
            transformed_image_sequence.append(image_transformed)

        return np.stack(transformed_image_sequence)

    def load_img(self, sequence_path, idx, bucket):
        full_image_path = sequence_path + "_{}.png".format(idx)
        try:
            blob = bucket.get_blob(full_image_path)
            if blob is None:
                log.info(f"Cannot get blob with path: {full_image_path}")
                return None
            blob = blob.download_as_string()
            blob_in_bytes = io.BytesIO(blob)
            img = np.asarray(Image.open(blob_in_bytes).convert("RGB"))
            return img
        except ServiceUnavailable:
            log.error("Service unavailable")
            return None




# Classifier

class DeepfakeClassifier3D(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.encoder = encoder_params_3D["tf_efficientnet_b7_ns"]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(dropout_rate)
        self.fc = Linear(encoder_params_3D["tf_efficientnet_b7_ns"]["features"], 1)

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


# Training and Testing

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
    minimum_loss = 1
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

    # load best model weights
    model.load_state_dict(torch.load(model_save_path))
    return model


if __name__ == '__main__':
    # Prepare logging
    from logging.handlers import RotatingFileHandler
    handler = RotatingFileHandler(filename='training_3D.log', maxBytes=20000000, backupCount=10)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s, %(name)s, %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[handler])
    log = logging.getLogger(__name__)

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
    num_workers = 8

    # Prepare buckets

    from google.cloud import storage

    train_dataset = DeepFakeDataset3D(sequence_dataframe, real_folder_to_identifiers, fake_folder_to_identifiers,
                                      augmentation_pipeline_3D())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn, drop_last=True)

    # need to create testing dataset
    testing_sequence_dataframe = pd.read_csv(SEQUENCE_DATAFRAME_TESTING_PATH)
    with open(TESTING_FOLDER_TO_IDENTIFIERS_PATH, 'rb') as handle:
        testing_folder_to_identifiers = pickle.load(handle)

    NAME_TEST = "faces_validation"

    testing_dataset = TestingDataset(testing_sequence_dataframe, validation_augmentation_pipeline(),
                                     testing_folder_to_identifiers)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size_testing, shuffle=False,
                                num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)

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
    optimizer_ft = optim.SGD(model.parameters(), lr=0.005, momentum=0.9,
                             weight_decay=1e-4, nesterov=True)
    lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.8)

    probability_distribution = distributions.beta.Beta(0.5, 0.5)

    log.info("Training Begins")

    history = {
        "train": [],
        "val": []
    }
    client = storage.Client()

    model = train_model(model, criterion, optimizer_ft, lr_scheduler, 25)
