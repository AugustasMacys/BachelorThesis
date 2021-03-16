"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""

import logging
import os
import os.path
import random

import numpy as np
import torch
import torch.utils.data as data

from coviar import get_num_frames
from coviar import load
from training_coviar.coviarTransforms import color_aug


GOP_SIZE = 12


log = logging.getLogger(__name__)


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


def get_seg_range(n, num_segments, seg, representation):
    if representation in ['residual', 'mv']:
        n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg+1)))
    if seg_end == seg_begin:
        seg_end = seg_begin + 1

    if representation in ['residual', 'mv']:
        # Exclude the 0-th frame, because it's an I-frame.
        return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


def get_gop_pos(frame_idx, representation):
    gop_index = frame_idx // GOP_SIZE
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_index -= 1
            gop_pos = GOP_SIZE - 1
    else:
        # I-frame
        gop_pos = 0
    return gop_index, gop_pos


class CoviarDataSet(data.Dataset):
    def __init__(self,
                 dataframe,
                 representation,
                 transform,
                 num_segments,
                 is_train,
                 accumulate):

        self.videos_dataframe = dataframe
        self._num_segments = num_segments
        self._representation = representation
        self._transform = transform
        self._is_train = is_train
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    # def _load_list(self, video_list):
    #     self._video_list = []
    #     with open(video_list, 'r') as f:
    #         for line in f:
    #             video, _, label = line.strip().split()
    #             video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
    #             self._video_list.append((
    #                 video_path,
    #                 int(label),
    #                 get_num_frames(video_path)))

        # print('%d videos loaded.' % len(self._video_list))

    def _get_train_frame_index(self, num_frames, seg):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                           representation=self._representation)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, self._representation)

    def _get_test_frame_index(self, num_frames, seg):
        if self._representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, self._representation)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        real_encoded_video = row["real_encoded_video"]
        fake_encoded_video = row["fake_encoded_video"]

        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 0

        num_frames_real = get_num_frames(real_encoded_video)
        num_frames_fake = get_num_frames(fake_encoded_video)
        num_frames = min(num_frames_real, num_frames_fake)
        # if self._is_train:
        #     video_path, label, num_frames = random.choice(self._video_list)
        # else:
        #     video_path, label, num_frames = self._video_list[index]

        real_frames = []
        fake_frames = []
        for seg in range(self._num_segments):

            # if self._is_train:
            gop_index, gop_pos = self._get_train_frame_index(num_frames, seg)

            # So far focus on training
            # else:
            #     gop_index, gop_pos = self._get_test_frame_index(num_frames, seg)

            img_real = load(real_encoded_video, gop_index, gop_pos,
                            representation_idx, self._accumulate)

            img_fake = load(fake_encoded_video, gop_index, gop_pos,
                            representation_idx, self._accumulate)

            if img_real is None:
                log.error('Error: loading video %s failed.' % real_encoded_video)
                img_real = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))

            if img_fake is None:
                log.error('Error: loading video %s failed.' % fake_encoded_video)
                img_fake = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))

            else:
                if self._representation == 'mv':
                    img_real = clip_and_scale(img_real, 20)
                    img_real += 128
                    img_real = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

                    img_fake = clip_and_scale(img_fake, 20)
                    img_fake += 128
                    img_fake = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                elif self._representation == 'residual':
                    img_real += 128
                    img_real = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

                    img_fake += 128
                    img_fake = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

            if self._representation == 'iframe':
                img_real = color_aug(img_real)
                # BGR to RGB. (PyTorch uses RGB according to doc.)
                img_real = img_real[..., ::-1]

                img_fake = color_aug(img_fake)
                # BGR to RGB. (PyTorch uses RGB according to doc.)
                img_fake = img_fake[..., ::-1]

            real_frames.append(img_real)
            fake_frames.append(img_fake)

        frames = self._transform(frames)

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        input = torch.from_numpy(frames).float() / 255.0

        if self._representation == 'iframe':
            input = (input - self._input_mean) / self._input_std
        elif self._representation == 'residual':
            input = (input - 0.5) / self._input_std
        elif self._representation == 'mv':
            input = (input - 0.5)

        return input, label

    def __len__(self):
        return len(self.videos_dataframe)
