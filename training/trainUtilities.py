import torch
from torchvision.transforms import Normalize

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

normalize_transform = Normalize(MEAN, STD)


class Unnormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        return torch.clamp(tensor*std + mean, 0., 1.)
