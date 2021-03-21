"""Run testing given a trained model."""
import argparse
import logging
import time

import pandas as pd
import numpy as np
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
import torchvision


from coviar import get_num_frames
from coviar import load

from training_coviar.dataset import clip_and_scale
from training_coviar.coviarModel import Model
from training_coviar.coviarTransforms import GroupCenterCrop
from training_coviar.coviarTransforms import GroupScale

GOP_SIZE = 12


log = logging.getLogger(__name__)


class CoviarCombinedTestDataSet(data.Dataset):
    def __init__(self,
                 dataframe,
                 transform,
                 num_segments,
                 accumulate):

        self.videos_dataframe = dataframe
        self._num_segments = num_segments
        self._transform = transform
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

    def __getitem__(self, index):
        row = self.videos_dataframe.iloc[index]
        encoded_video = row["encoded_video"]
        label = row["label"]

        mv_representation_idx = 1
        residual_representation_idx = 2
        iframe_representation_idx = 0

        num_frames = get_num_frames(encoded_video)

        frames = []
        vectors = []
        residuals = []
        for seg in range(self._num_segments):

            num_frames_mv_residual = num_frames - 1

            seg_size_iframe = float(num_frames - 1) / self._num_segments
            seg_size_mv_residual = float(num_frames_mv_residual - 1) / self._num_segments
            v_frame_idx_iframe = int(np.round(seg_size_iframe * (seg + 0.5)))
            v_frame_idx_mv_residual = int(np.round(seg_size_mv_residual * (seg + 0.5))) + 1

            gop_index_mv_residual = (v_frame_idx_mv_residual // GOP_SIZE)
            gop_index_iframe = v_frame_idx_iframe // GOP_SIZE
            gop_pos_mv_residual = (gop_index_mv_residual % GOP_SIZE)
            gop_pos_iframe = 0

            if gop_pos_mv_residual == 0:
                gop_index_mv_residual -= 1
                gop_pos_mv_residual = GOP_SIZE - 1

            motion_vector = load(encoded_video, gop_index_mv_residual, gop_pos_mv_residual,
                                 mv_representation_idx, self._accumulate)

            residual = load(encoded_video, gop_index_mv_residual, gop_pos_mv_residual,
                            residual_representation_idx, self._accumulate)

            iframe = load(encoded_video, gop_index_iframe, gop_pos_iframe,
                          iframe_representation_idx, self._accumulate)

            if motion_vector is None:
                log.error('Error: loading motion vector %s failed.' % encoded_video)
                motion_vector = np.zeros((256, 256, 2)).astype('uint8')

            if residual is None:
                log.error('Error: loading residual %s failed.' % encoded_video)
                residual = np.zeros((256, 256, 3)).astype('uint8')

            if iframe is None:
                log.error('Error: loading iframe %s failed.' % encoded_video)
                iframe = np.zeros((256, 256, 3)).astype('uint8')

            else:
                motion_vector = clip_and_scale(motion_vector, 20)
                motion_vector += 128
                motion_vector = (np.minimum(np.maximum(motion_vector, 0), 255)).astype(np.uint8)

                residual += 128
                residual = (np.minimum(np.maximum(residual, 0), 255)).astype(np.uint8)

                iframe = iframe[..., ::-1]

            frames.append(iframe)
            residuals.append(residual)
            vectors.append(motion_vector)

        frames = self._transform(frames)
        residuals = self._transform(frames)
        vectors = self._transform(frames)

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))

        residuals = np.array(residuals)
        residuals = np.transpose(residuals, (0, 3, 1, 2))

        vectors = np.array(vectors)
        vectors = np.transpose(vectors, (0, 3, 1, 2))

        input_frames = torch.from_numpy(frames).float() / 255.0
        input_residuals = torch.from_numpy(residuals).float() / 255.0
        input_vectors = torch.from_numpy(vectors).float() / 255.0

        input_frames = (input_frames - self._input_mean) / self._input_std
        input_residuals = (input_residuals - 0.5) / self._input_std
        input_vectors = (input_vectors - 0.5)

        combination = {
            "frame": input_frames,
            "residual": input_residuals,
            "vector": input_vectors,
            "label": label
        }

        return combination

    def __len__(self):
        return len(self.videos_dataframe)


def apply_shift(outputs):
    delta = outputs - 0.5
    sign_array = np.sign(delta)
    pos_array = delta > 0
    neg_array = delta < 0
    outputs[pos_array] = np.clip(0.5 + sign_array[pos_array] * np.power(abs(delta[pos_array]),
                                                                               0.65), 0.01, 0.99)
    outputs[neg_array] = np.clip(0.5 + sign_array[neg_array] * np.power(abs(delta[neg_array]),
                                                                               0.65), 0.01, 0.99)

    weights = np.power(abs(delta), 1.0) + 1e-4
    final_score = float((outputs * weights).sum() / weights.sum())
    return final_score


if __name__ == '__main__':
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Standard video-level testing")
    parser.add_argument('--no-accumulation', action='store_true',
                        help='disable accumulation of motion vectors and residuals.')
    parser.add_argument('--dataframe', type=str)
    parser.add_argument('--modelIframe', type=str)
    parser.add_argument('--modelVector', type=str)
    parser.add_argument('--modelResidual', type=str)
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of workers for data loader.')

    parser.add_argument('--representation', type=str, choices=['iframe', 'mv', 'residual'],
                        help='data representation.')
    parser.add_argument('--arch', type=str, default="resnet34",
                        help='base architecture.')
    parser.add_argument('--num_segments', type=int, default=3,
                        help='number of TSN segments.')

    args = parser.parse_args()

    num_classes = 1

    model_motion_vector = Model(num_classes, args.num_segments, 'mv', args.arch)
    model_motion_vector.to(gpu)
    model_motion_vector = model_motion_vector.load_state_dict(torch.load(args.modelVector), strict=True)

    model_residual = Model(num_classes, args.num_segments, 'residual', args.arch)
    model_residual.to(gpu)
    model_residual = model_residual.load_state_dict(torch.load(args.modelResidual), strict=True)

    model_frame = Model(num_classes, args.num_segments, 'iframe')
    model_frame.to(gpu)
    model_frame = model_frame.load_state_dict(torch.load(args.modelIframe), strict=True)

    model_frame.eval()
    model_residual.eval()
    model_motion_vector.eval()

    log.info("Models are loaded")

    testing_dataframe = pd.read_csv(args.dataframe)

    data_loader = data.DataLoader(
        CoviarCombinedTestDataSet(
            testing_dataframe,
            num_segments=args.num_segments,
            transform=torchvision.transforms.Compose([
                # all the same
                GroupScale(int(model_frame.scale_size)),
                GroupCenterCrop(model_frame.crop_size),
            ]),
            accumulate=(not args.no_accumulation),
        ),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    total_num = len(data_loader.dataset)
    outputs = []
    labels = []

    for combination in data_loader:
        label = combination["label"].to(gpu)
        vector = combination["vector"].to(gpu)
        frame = combination["frame"].to(gpu)
        residual = combination["residual"].to(gpu)

        output_vector = model_motion_vector(vector)
        output_vector = output_vector.view((-1, args.num_segments) + output_vector.size()[1:])
        output_vector = apply_shift(output_vector)
        # output_vector = torch.mean(output_vector, dim=1)

        output_residual = model_residual(residual)
        output_residual = output_residual.view((-1, args.num_segments) + output_residual.size()[1:])
        output_residual = apply_shift(output_residual)
        # output_residual = torch.mean(output_residual, dim=1)

        output_frame = model_residual(residual)
        output_frame = output_residual.view((-1, args.num_segments) + output_frame.size()[1:])
        output_frame = apply_shift(output_frame)
        # output_frame = torch.mean(output_frame, dim=1)

        mean_outputs = (output_vector + output_residual + output_frame) / 3

        outputs.append(mean_outputs)
        labels.append(label.item())

    df = pd.DataFrame(list(zip(outputs, labels)),
                      columns=['Prediction', 'Truth'])

    df.to_csv("prediction_dataframe.csv", index=False)
