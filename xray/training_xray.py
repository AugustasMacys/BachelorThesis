import segmentation_models_pytorch as smp

import torch
import logging


log = logging.getLogger(__name__)


if __name__ == '__main__':
    gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    segmentation_model = smp.DeepLabV3Plus(
        encoder_name="timm-efficientnet-b4",
        encoder_depth=3,
        encoder_weights="noisy-student",
        classes=2,
        in_channels=1
    )

    log.info("Model is initialised")

    history = {
        "train": [],
        "val": []
    }