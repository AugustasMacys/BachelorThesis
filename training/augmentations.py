import cv2

from torchvision import transforms
from albumentations import (
    HorizontalFlip, Blur, GaussianBlur, HueSaturationValue, DualTransform,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur,
    IAASharpen, IAAEmboss, OneOf, Compose, RandomBrightnessContrast, ToSepia, ImageCompression, ShiftScaleRotate
)

from training.trainUtilities import MEAN, STD


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


def augmentation_pipeline(p=0.5, size=320):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        HorizontalFlip(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.25),
            GaussianBlur(p=0.5),
            Blur(blur_limit=3, p=0.25),
        ], p=0.2),
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        HueSaturationValue(p=0.2),
        OneOf([
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.6),
        ToSepia(p=0.1),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ], p=p)


transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
