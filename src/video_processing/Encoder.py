import ntpath
import os
import subprocess
from tqdm import tqdm


train_folder_path = r"D:/deepfakes/data/train"
test_folder_path = r"D:/deepfakes/data/test"


VIDEO_WIDTH = 192
VIDEO_HEIGHT = 224


def encode_mpeg4(video_path):
    video_name = ntpath.basename(video_path)[:-4]
    directory_to_save = os.path.dirname(video_path)
    output_name = video_name + "_" + "mpeg4" + ".mp4"
    output_path = os.path.join(directory_to_save, output_name)
    subprocess.call(["ffmpeg", "-i", video_path, "-vf",
                     "scale=" + str(VIDEO_WIDTH) + ":" + str(VIDEO_HEIGHT) + ",setsar=1:1",
                     "-q:v", "1", "-c:v", "mpeg4", "-f", "rawvideo", output_path])


if __name__ == '__main__':
    videos = []
    for subdir, dirs, files in os.walk(test_folder_path):
        for file in files:
            if file.endswith(".mp4") and "mpeg" not in file:
                videos.append(os.path.join(subdir, file))

    with tqdm(total=len(videos)) as bar:
        for video_path in videos:
            encode_mpeg4(video_path)
            os.remove(video_path)

            bar.update()
