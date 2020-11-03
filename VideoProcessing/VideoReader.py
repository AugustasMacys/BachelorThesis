import argparse
import cv2
import glob
import ntpath
import os

from functools import partial
from multiprocessing.pool import Pool
from tqdm import tqdm

from utilities import ROOT_DIR


def video_frame_extractor(video_name):
    capturator = cv2.VideoCapture(video_name)
    frames_number = capturator.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in range(frames_number):
        capturator.grab()
        if i % 5 == 0:
            result, frame = capturator.retrieve()
            if not result:
                continue

        identifier = ntpath.basename(video_name) + '_' + str(i)
        cv2.imwrite(os.path.join(ROOT_DIR, "extracted_images", identifier), [cv2.IMWRITE_JPEG_QUALITY, 100])

    capturator.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--videoFolder")

    args = parser.parse_args()

    full_video_path = os.path.join(ROOT_DIR, args.videoFolder)
    video_filenames = [path for path in glob.glob(os.path.join(full_video_path, "*/*.mp4"))]
    with Pool(processes=os.cpu_count() - 2) as pool:
        with tqdm(total=len(video_filenames)) as bar:
            for video in pool.imap_unordered(partial(video_frame_extractor), video_filenames):
                bar.update()
