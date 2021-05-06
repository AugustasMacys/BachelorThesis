import os
from glob import glob

import cv2
import insightface
import ntpath
from tqdm import tqdm


from src.Utilities import VALIDATION_DIRECTORY, VALIDATION_FACES_DIRECTORY


def video_frame_extractor(video_name, folder):
    capturator = cv2.VideoCapture(video_name)
    frames_number = int(capturator.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capturator.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = 1.0
    for i in range(frames_number):
        capturator.grab()
        if i % 20 == 0: # Depends how many frames I want
            success, frame = capturator.retrieve()

            if not success:
                continue

            if width <= 300:
                scale = 2.0
            elif 1000 < width <= 1900:
                scale = 0.5
            elif width > 1900:
                scale = 0.33

            bounding_box, _ = model.detect(frame, threshold=0.7, scale=scale)
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

            identifier = ntpath.basename(video_name)[:-4] + '_' + str(i)
            directory_to_create = os.path.join(folder, ntpath.basename(video_name)[:-4])
            if not os.path.exists(directory_to_create):
                os.mkdir(directory_to_create)
            save_string = os.path.join(directory_to_create, identifier) + ".png"
            cv2.imwrite(save_string, frame)

    capturator.release()


if __name__ == '__main__':
    output_folder = VALIDATION_DIRECTORY
    videos = glob(os.path.join(VALIDATION_DIRECTORY, "*.mp4"))
    model = insightface.model_zoo.get_model('retinaface_r50_v1')
    model.prepare(ctx_id=0, nms=0.4)

    output_folder = VALIDATION_FACES_DIRECTORY
    with tqdm(total=len(videos)) as bar:
        for video in videos:
            video_frame_extractor(video, output_folder)
            bar.update()
