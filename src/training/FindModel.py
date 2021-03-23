import timm
from pprint import pprint
from torchsummary import summary
import torch

from albumentations.pytorch import ToTensorV2

import numpy as np

# m = timm.create_model('tf_efficientnet_l2_ns_475', pretrained=True).cuda()
# print(m._fc.in_features)
# print(m.num_features)
# summary(m, (3, 224, 224))

# t = torch.tensor([[[1, 2],
#                   [3, 4]],
#                   [[5, 6],
#                   [7, 8]]])
#
# print(t.shape)
# t = t.flatten(1)
# print(t.shape)
# print(t)

# x = torch.randn(2, 3)
# x = [torch.stack([x])]
# a = torch.cat(x)
# print(a.shape)

# from training.augmentations import transformation
#
# track_sequences = []
# sequence = [np.zeros((224, 224, 3)) for x in range(10)]
# for i in range(5):
#     track_sequences.append(sequence)
#
#
# track_sequences = [torch.stack([transformation(face) for face in sequence]) for sequence in track_sequences]
# track_sequences = torch.cat(track_sequences)
# print(track_sequences.shape)


from coviar import load, get_num_frames
path_video = r"D:\deepfakes\special_video_change_format\output6.mp4"
# path_video = r"D:\deepfakes\special_video_change_format\a.mp4"
# b = get_num_frames(path_video)
# print(b)
# a = load(path_video, 1, 1, 2, True)


