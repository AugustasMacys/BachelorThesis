import cv2


def video_frame_extractor(video_name):

    capturator = cv2.VideoCapture(video_name)
    frames_number = capturator.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in range(frames_number):
        capturator.grab()
        if i % 5 == 0:
            result, frame = capturator.retrieve()
            if not result:
                continue
