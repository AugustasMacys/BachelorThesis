from collections import defaultdict
import os
from glob import glob
import time


import cv2
import insightface
import numpy as np
from PIL import Image
from timm.models.efficientnet_blocks import InvertedResidual
import torch
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity


from src.training.Augmentations import isotropically_resize_image, put_to_center, transformation
from src.training.TrainModelFaces2D import DeepfakeClassifier
from src.Utilities import MODELS_DIECTORY, VALIDATION_DIRECTORY
from src.training.TrainModelFaces3D import DeepfakeClassifier3D_V3, ConvolutionExpander


testing_model_path = os.path.join(MODELS_DIECTORY, "lowest_loss_model_new_heavy_augmentations_newer.pth")

scores_path = "efficient_net2D.csv"
final_path = "final_efficient_net2D.csv"


class InferenceLoader:

    def __init__(self, video_dir, face_detector,
                 transform=None, batch_size=15, face_limit=15, dimensions_3D=False):
        self.video_dir = video_dir
        self.test_videos = sorted([y for x in os.walk(self.video_dir) for y in glob(
            os.path.join(x[0], '*.mp4'))])

        self.transform = transform
        self.face_detector = face_detector

        self.batch_size = batch_size
        self.face_limit = face_limit

        self.record = defaultdict(list)
        self.score = defaultdict(lambda: 0.5)
        self.feedback_queue = []

        self.need_resize = dimensions_3D

    def iter_one_face(self):
        for file_name in self.test_videos:

            capturator = cv2.VideoCapture(file_name)
            frames_number = int(capturator.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capturator.get(cv2.CAP_PROP_FRAME_WIDTH))
            scale = 1.0
            face_counter = 0

            for i in range(frames_number):
                capturator.grab()
                if i % 20 == 0:
                    success, frame = capturator.retrieve()
                    if not success:
                        continue

                    if width <= 300:
                        scale = 2.0
                    elif 1000 < width <= 1900:
                        scale = 0.5
                    elif width > 1900:
                        scale = 0.33

                    bounding_box, landmarks = face_detector.detect(frame, threshold=0.7, scale=scale)
                    if bounding_box.shape[0] == 0:
                        # make the image brighter
                        yen_threshold = threshold_yen(frame)
                        frame = rescale_intensity(frame, (0, yen_threshold), (0, 255))
                        bounding_box, landmarks = face_detector.detect(frame, threshold=0.55, scale=scale)
                        # if again nothing is found just move on
                        if bounding_box.shape[0] == 0:
                            continue

                    x_min = bounding_box[:, 0]
                    y_min = bounding_box[:, 1]
                    x_max = bounding_box[:, 2]
                    y_max = bounding_box[:, 3]

                    areas = (y_max - y_min) * (x_max - x_min)
                    max_face_idx = areas.argmax()

                    w = x_max[max_face_idx] - x_min[max_face_idx]
                    h = y_max[max_face_idx] - y_min[max_face_idx]

                    margin_width = w // 4
                    margin_height = h // 4

                    frame = frame[max(int(y_min[max_face_idx] - margin_height), 0):int(y_max[max_face_idx] + margin_height),
                            max(int(x_min[max_face_idx] - margin_width), 0):int(x_max[max_face_idx] + margin_width)]

                    resized_frame = isotropically_resize_image(frame, 224)
                    resized_frame = put_to_center(resized_frame, 224)
                    if self.need_resize:
                        resized_frame = cv2.resize(resized_frame, (192, 224))
                    transformed_image = Image.fromarray(resized_frame[:, :, ::-1])

                    # normalise and apply prediction transform
                    if self.transform:
                        transformed_image = self.transform(transformed_image)

                    yield file_name, transformed_image

                    face_counter += 1
                    if face_counter == self.face_limit:
                        break

            capturator.release()

    def __iter__(self):
        self.record.clear()
        self.feedback_queue.clear()

        batch_buf = []
        t0 = time.time()
        batch_count = 0

        for file_name, transformed_face in self.iter_one_face():
            self.feedback_queue.append(file_name)
            batch_buf.append(transformed_face)
            if len(batch_buf) == self.batch_size:
                yield torch.stack(batch_buf)

                batch_count += 1
                batch_buf.clear()

        if len(batch_buf) > 0:
            yield torch.stack(batch_buf)

    def feedback(self, pred):
        accessed = set()

        for score in pred:
            file_name = self.feedback_queue.pop(0)
            accessed.add(file_name)
            self.record[file_name].append(score)

        for file_name in sorted(accessed):
            outgoing_score = np.array(self.record[file_name])

            delta = outgoing_score - 0.5
            sign_array = np.sign(delta)
            pos_array = delta > 0
            neg_array = delta < 0
            outgoing_score[pos_array] = np.clip(0.5 + sign_array[pos_array] * np.power(abs(delta[pos_array]),
                                                                                       0.65), 0.01, 0.99)
            outgoing_score[neg_array] = np.clip(0.5 + sign_array[neg_array] * np.power(abs(delta[neg_array]),
                                                                                       0.65), 0.01, 0.99)

            weights = np.power(abs(delta), 1.0) + 1e-4
            final_score = float((outgoing_score * weights).sum() / weights.sum())
            self.score[file_name] = final_score

        # Write just outgoing results just in case of CUDA: Memory Out of Space Error
        with open(scores_path, "w") as f:
            for key in self.score.keys():
                f.write("%s,%s\n" % (key, self.score[key]))


if __name__ == '__main__':
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3D model
    # model = DeepfakeClassifier3D_V3()
    # SEQUENCE_LENGTH = 5
    #
    # for module in model.modules():
    #     if isinstance(module, InvertedResidual):
    #         if module.exp_ratio != 1.0:
    #             expansion_con = module.conv_pw
    #             expander = ConvolutionExpander(expansion_con.in_channels, expansion_con.out_channels, SEQUENCE_LENGTH)
    #             # 5 dimension tensor and we take third dimension
    #             expander.conv.weight.data[:, :, 0, :, :].copy_(expansion_con.weight.data / 3)
    #             expander.conv.weight.data[:, :, 1, :, :].copy_(expansion_con.weight.data / 3)
    #             expander.conv.weight.data[:, :, 2, :, :].copy_(expansion_con.weight.data / 3)
    #             module.conv_pw = expander
    #
    # pretrained_weights_path = r"D:\deepfakes\trained_models\3Dnew_model5.pth"
    # model.load_state_dict(torch.load(pretrained_weights_path))
    # model.to(gpu)

    # uncomment for 2D model
    model = DeepfakeClassifier()
    model.to(gpu)
    model.load_state_dict(torch.load(testing_model_path))
    model.eval()

    validation_directory = VALIDATION_DIRECTORY
    face_detector = insightface.model_zoo.get_model('retinaface_r50_v1')
    face_detector.prepare(ctx_id=0, nms=0.4)
    loader = InferenceLoader(validation_directory, face_detector, transformation)

    for batch in loader:
        batch = batch.cuda(non_blocking=True)
        with torch.no_grad():
            y_pred = model(batch)
            y_pred = torch.sigmoid(y_pred.squeeze())
            loader.feedback(y_pred)

    with open(final_path, "w") as f:
        for key in loader.score.keys():
            f.write("%s,%s\n" % (key, loader.score[key]))
