import os
import time
from collections import defaultdict

import cv2
import numpy as np

import insightface
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import transforms as T


from utilities import MODELS_DIECTORY
from VideoProcessing.VideoReader import norm_crop, ARCFACE_REFERENCE

model_save_path = os.path.join(MODELS_DIECTORY, "lowest_loss_model.pth")

scores_path = "scores.csv"


class InferenceLoader:

    def __init__(self, video_dir, face_detector,
                 transform=None, batch_size=15, face_limit=15):
        self.video_dir = video_dir
        self.test_videos = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))[:5]

        self.transform = transform
        self.face_detector = face_detector

        self.batch_size = batch_size
        self.face_limit = face_limit

        self.record = defaultdict(list)
        self.score = defaultdict(lambda: 0.5)
        self.feedback_queue = []

    def iter_one_face(self):
        for file_name in self.test_videos:
            full_path = os.path.join(self.video_dir, file_name)

            capturator = cv2.VideoCapture(full_path)
            frames_number = int(capturator.get(cv2.CAP_PROP_FRAME_COUNT))
            face_counter = 0

            for i in range(frames_number):
                capturator.grab()
                if i % 20 == 0:
                    success, frame = capturator.retrieve()
                    if not success:
                        continue

                    print(file_name)
                    bounding_box, landmarks = face_detector.detect(frame, threshold=0.5, scale=1.0)
                    if bounding_box.shape[0] == 0:
                        continue

                    areas = (bounding_box[:, 3] - bounding_box[:, 1]) * (bounding_box[:, 2] - bounding_box[:, 0])
                    max_face_idx = areas.argmax()
                    face_landmark = landmarks[max_face_idx]

                    face_landmark = face_landmark.reshape(5, 2).astype(np.int)
                    transformed_image = norm_crop(frame, face_landmark, ARCFACE_REFERENCE, image_size=224)
                    transformed_image = Image.fromarray(transformed_image[:, :, ::-1])

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

        for file_name, face in self.iter_one_face():
            self.feedback_queue.append(file_name)
            batch_buf.append(face)

            if len(batch_buf) == self.batch_size:
                yield torch.stack(batch_buf)

                batch_count += 1
                batch_buf.clear()

                if batch_count % 15 == 0:
                    elapsed = 1000 * (time.time() - t0)
                    print("T: %.2f ms / batch" % (elapsed / batch_count))

        if len(batch_buf) > 0:
            yield torch.stack(batch_buf)

    def feedback(self, pred):
        accessed = set()

        for score in pred:
            file_name = self.feedback_queue.pop(0)
            accessed.add(file_name)
            self.record[file_name].append(score)

        for file_name in sorted(accessed):
            self.score[file_name] = torch.mean(torch.stack(self.record[file_name]), dim=0)
            print("[%s] %.6f" % (file_name, self.score[file_name]))


if __name__ == '__main__':
    gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EfficientNet.from_name('efficientnet-b0', num_classes=1)
    model._fc = nn.Linear(1280, 1)
    model.to(gpu)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    validation_directory = r"D:\deepfakes\data\test"
    face_detector = insightface.model_zoo.get_model('retinaface_r50_v1')
    face_detector.prepare(ctx_id=0, nms=0.4)
    loader = InferenceLoader(validation_directory, face_detector, T.ToTensor())

    for batch in loader:
        batch = batch.cuda(non_blocking=True)
        with torch.no_grad():
            y_pred = model(batch)
            y_pred = torch.sigmoid(y_pred.squeeze())
            loader.feedback(y_pred)

    with open(scores_path, "w") as f:
        for key in loader.score.keys():
            f.write("%s,%s\n" % (key, loader.score[key].item()))
