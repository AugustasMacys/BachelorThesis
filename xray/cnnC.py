import os
import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class CNNC(nn.Module):
    # Definition in https://arxiv.org/pdf/1912.13458.pdf

    def __init__(self):
        super(CNNC, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1 * 1 * 1, 2)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def predict(self, x):
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)

        return probabilities

    def init_weights(self, pretrained='',):
        log.info('Init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            if torch.cuda.is_available():
                pretrained_dict = torch.load(pretrained)
            else:
                pretrained_dict = torch.load(pretrained, map_location='cpu')
            log.info('Loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                log.info('Loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            for k, v in self.named_parameters():
                if 'upsample_modules' in k or 'output_modules' in k:
                    continue
                else:
                    v.requires_grad = False


def get_nnc(config, **kwargs):
    model = CNNC()
    model.init_weights(pretrained=config.TEST.NNC_FILE)
    return model