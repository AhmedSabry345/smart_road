import cv2

class VideoReader:
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.cap = cv2.VideoCapture(self.source_path)

        # handles the video if it is not opend
        if not self.cap.isOpened():
            raise ValueError(f"Error: unable to open the video")


        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # getting the frames
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()