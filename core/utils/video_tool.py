import os
import cv2


class VideoMaker:
    def __init__(self, frames_dir, out_dir, fps=30, img_size=(854, 480)):
        self.frames_dir = frames_dir
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        # fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.videowriter = cv2.VideoWriter(out_dir, fourcc, fps, img_size)

    def make_video(self):
        frames = sorted(os.listdir(self.frames_dir))
        for frame in frames:
            f_path = os.path.join(self.frames_dir, frame)
            image = cv2.imread(f_path)
            self.videowriter.write(image)

    def release(self):
        self.videowriter.release()
