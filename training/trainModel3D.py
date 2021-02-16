from functools import partial
import config_logger
import logging

from timm.models.efficientnet import tf_efficientnet_l2_ns_475
from torch.nn import functional as F, Dropout, Linear, AdaptiveAvgPool3d
import torch
import torch.nn as nn

from timm.models.efficientnet_blocks import InvertedResidual
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


SEQUENCE_LENGTH = 5


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

    def __init__(self, sequence_dataframe, augmentations, image_size=224):

        self.image_size = image_size
        if 'index' in sequence_dataframe:
            del sequence_dataframe['index']

        self.augmentate = augmentations
        self.df = sequence_dataframe
        self.identifiers = [60, 80, 100, 120, 140]
        self.emergency_identifiers = [0, 20, 40, 60, 80]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        real_image_folder = row["real_image_folder"]
        fake_image_folder = row["fake_image_folder"]

        # Will go here from second epoch
        if real_image_name in self.non_existing_files or fake_image_name in self.non_existing_files:
            return None

        try:
            img_real = Image.open(real_image_name).convert("RGB")
        except FileNotFoundError:
            log.info("Real Image not found: {}".format(real_image_name))
            return None
        try:
            img_fake = Image.open(fake_image_name).convert("RGB")
        except FileNotFoundError:
            log.info("Fake Image not found: {}".format(fake_image_name))
            return None

        img_real = np.array(img_real)
        img_fake = np.array(img_fake)
        transformed_images = self.augmentate(image=img_real, image2=img_fake)
        img_real = transformed_images["image"]
        img_fake = transformed_images["image2"]
        img_fake = F.resize(img_fake, height=224, width=224)
        gaussian_transformed_images = gaussian_noise_transform(image=img_real,
                                                               image2=img_fake)
        img_real = gaussian_transformed_images["image"]
        img_fake = gaussian_transformed_images["image2"]
        # print(img_real.shape)
        # print(img_fake.shape)
        # cv2.imwrite("test_real.png", img_real)
        # cv2.imwrite("test_fake.png", img_fake)
        img_real = img_to_tensor(img_real, {"mean": MEAN,
                                            "std": STD})
        img_fake = img_to_tensor(img_fake, {"mean": MEAN,
                                            "std": STD})

        pair = {
            "real": img_real,
            "fake": img_fake
        }

        return pair



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


