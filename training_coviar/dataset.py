"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""

import logging
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
                 accumulate):

        self.videos_dataframe = dataframe
        self._num_segments = num_segments
        self._representation = representation
        self._transform = transform
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

    def _get_train_frame_index(self, num_frames, seg):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                           representation=self._representation)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end - 1)
        return get_gop_pos(v_frame_idx, self._representation)

    def __getitem__(self, index):
        row = self.videos_dataframe.iloc[index]
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

        real_frames = []
        fake_frames = []
        for seg in range(self._num_segments):
            gop_index, gop_pos = self._get_train_frame_index(num_frames, seg)

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
                    img_real = (np.minimum(np.maximum(img_real, 0), 255)).astype(np.uint8)

                    img_fake = clip_and_scale(img_fake, 20)
                    img_fake += 128
                    img_fake = (np.minimum(np.maximum(img_fake, 0), 255)).astype(np.uint8)
                elif self._representation == 'residual':
                    img_real += 128
                    img_real = (np.minimum(np.maximum(img_real, 0), 255)).astype(np.uint8)

                    img_fake += 128
                    img_fake = (np.minimum(np.maximum(img_fake, 0), 255)).astype(np.uint8)

            if self._representation == 'iframe':
                # BGR to RGB. (PyTorch uses RGB according to doc.)
                prev_state = random.getstate()
                img_real = color_aug(img_real)
                img_real = img_real[..., ::-1]

                random.setstate(prev_state)
                img_fake = color_aug(img_fake)
                img_fake = img_fake[..., ::-1]

            real_frames.append(img_real)
            fake_frames.append(img_fake)

        # Comeback here
        prev_state = random.getstate()
        real_frames = self._transform(real_frames)

        random.setstate(prev_state)
        fake_frames = self._transform(fake_frames)

        real_frames = np.array(real_frames)
        real_frames = np.transpose(real_frames, (0, 3, 1, 2))

        fake_frames = np.array(fake_frames)
        fake_frames = np.transpose(fake_frames, (0, 3, 1, 2))

        real_input = torch.from_numpy(real_frames).float() / 255.0
        fake_input = torch.from_numpy(fake_frames).float() / 255.0

        if self._representation == 'iframe':
            real_input = (real_input - self._input_mean) / self._input_std
            fake_input = (fake_input - self._input_mean) / self._input_std
        elif self._representation == 'residual':
            real_input = (real_input - 0.5) / self._input_std
            fake_input = (fake_input - 0.5) / self._input_std
        elif self._representation == 'mv':
            real_input = (real_input - 0.5)
            fake_input = (fake_input - 0.5)

        log.info("pair returned successfully")

        pairs = {
            "fake": fake_input,
            "real": real_input
        }

        return pairs

    def __len__(self):
        return len(self.videos_dataframe)


class CoviarTestDataSet(data.Dataset):
    def __init__(self,
                 dataframe,
                 representation,
                 transform,
                 num_segments,
                 accumulate):

        self.videos_dataframe = dataframe
        self._num_segments = num_segments
        self._representation = representation
        self._transform = transform
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

    def _get_test_frame_index(self, num_frames, seg):
        if self._representation in ['mv', 'residual']:
            num_frames -= 1

        seg_size = float(num_frames - 1) / self._num_segments
        v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, self._representation)

    def __getitem__(self, index):
        row = self.videos_dataframe.iloc[index]
        encoded_video = row["encoded_video"]
        label = row["label"]

        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 0

        num_frames = get_num_frames(encoded_video)

        frames = []
        for seg in range(self._num_segments):
            gop_index, gop_pos = self._get_test_frame_index(num_frames, seg)

            img = load(encoded_video, gop_index, gop_pos,
                       representation_idx, self._accumulate)

            if img is None:
                log.error('Error: loading video %s failed.' % encoded_video)
                img = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))

            else:
                if self._representation == 'mv':
                    img = clip_and_scale(img, 20)
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

                elif self._representation == 'residual':
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

            if self._representation == 'iframe':
                # BGR to RGB. (PyTorch uses RGB according to doc.)
                img = color_aug(img)
                img = img[..., ::-1]

            frames.append(img)

        frames = self._transform(frames)

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        input_tensor = torch.from_numpy(frames).float() / 255.0

        if self._representation == 'iframe':
            input_tensor = (input_tensor - self._input_mean) / self._input_std
        elif self._representation == 'residual':
            input_tensor = (input_tensor - 0.5) / self._input_std
        elif self._representation == 'mv':
            input_tensor = (input_tensor - 0.5)

        pairs = {
            "input": input_tensor,
            "label": label
        }

        return pairs

    def __len__(self):
        return len(self.videos_dataframe)
