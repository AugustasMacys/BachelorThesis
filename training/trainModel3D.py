from functools import partial
import config_logger
import logging

from timm.models.efficientnet import tf_efficientnet_l2_ns_475
from torch.nn import functional as F, Dropout, Linear, AdaptiveAvgPool3d
import torch
import torch.nn as nn

from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch import EfficientNet
from timm.models.efficientnet_blocks import InvertedResidual

log = logging.getLogger(__name__)


encoder_params_3D = {
    "tf_efficientnet_l2_ns_475": {
        "features": 5504,
        "init_op": partial(tf_efficientnet_l2_ns_475,
                           num_classes=1,
                           pretrained=True,
                           drop_rate=0.5)
    }
}


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
    def __init__(self, in_channels, out_channels, seq_length):
        super(ConvolutionExpander, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.seq_length = seq_length

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        x = x.view(batch_size // self.seq_length, self.seq_length, in_channels, height, width)
        x = self.conv(x.transpose(1, 2).contiguous()).transpose(2, 1).contiguous()
        x = x.flatten(0, 1)
        return x


if __name__ == '__main__':
    # gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # log.info("Program Started")
    # log.info(f"GPU value: {gpu}")

    ## create dataloaders

    # log.info(f"Dataloaders Created")

    model = DeepfakeClassifier3D()

    timm_changes = 0
    timm_residuals = 0
    for module in model.modules():
        if isinstance(module, InvertedResidual):
            timm_residuals += 1
            print(module.exp_ratio)
            if module.exp_ratio != 1.0:
                expand_conv = module.conv_pw
                timm_changes += 1
                # seq_expand_conv = SeqExpandConv(expand_conv.in_channels, expand_conv.out_channels, len(TRAIN_INDICES))
                # seq_expand_conv.conv.weight.data[:, :, 0, :, :].copy_(expand_conv.weight.data / 3)
                # seq_expand_conv.conv.weight.data[:, :, 1, :, :].copy_(expand_conv.weight.data / 3)
                # seq_expand_conv.conv.weight.data[:, :, 2, :, :].copy_(expand_conv.weight.data / 3)
                # module._expand_conv = seq_expand_conv

    # log.info("Model is initialised")
    print(timm_changes)
    print(timm_residuals)

    model = EfficientNet.from_name('efficientnet-b7', override_params={'num_classes': 1})

    effnet_changes = 0
    effnet_residuals = 0
    for module in model.modules():
        if isinstance(module, MBConvBlock):
            effnet_residuals += 1
            print(module._block_args.expand_ratio)
            if module._block_args.expand_ratio != 1:
                expand_conv = module._expand_conv
                effnet_changes += 1
                # seq_expand_conv = SeqExpandConv(expand_conv.in_channels, expand_conv.out_channels, len(TRAIN_INDICES))
                # seq_expand_conv.conv.weight.data[:, :, 0, :, :].copy_(expand_conv.weight.data / 3)
                # seq_expand_conv.conv.weight.data[:, :, 1, :, :].copy_(expand_conv.weight.data / 3)
                # seq_expand_conv.conv.weight.data[:, :, 2, :, :].copy_(expand_conv.weight.data / 3)
                # module._expand_conv = seq_expand_conv

    # log.info("Model is initialised")
    print(effnet_changes)
    print(effnet_residuals)

