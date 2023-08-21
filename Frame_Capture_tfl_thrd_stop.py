import cv2  # For Image processing
import threading
from tflite_support.task import core, processor, vision
import os


class piVideoStream(threading.Thread):

    def __init__(self, resolution=(1280, 720), framerate=30):
        super().__init__()
        self.stop_flag = threading.Event()

        self.full_path = os.path.realpath(__file__)
        self.WD = os.path.dirname(self.full_path)

        self.model = self.WD + '/efficientdet_lite0.tflite'
        self.num_threads = 4
        
        self.frame = []

        self.cam = cv2.VideoCapture(0)  # Get video feed from the Camera
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])  # set video widht
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])  # set video height
        self.cam.set(cv2.CAP_PROP_FPS, framerate)

        # Define min window size to be recognized as a face
        self.minW = 0.1 * self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.minH = 0.1 * self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.ret = None

        self.base_options=core.BaseOptions(file_name=self.model,use_coral=False,num_threads=self.num_threads)
        self.detection_options=processor.DetectionOptions(max_results=8, score_threshold=.3)
        self.options=vision.ObjectDetectorOptions(base_options=self.base_options,detection_options=self.detection_options)
        self.detector=vision.ObjectDetector.create_from_options(self.options)

    # def start(self):
    #     # start the thread to read frames from the video stream
    #     self.thread = Thread(target=self.update, args=())
    #     self.thread.daemon = True
    #     self.thread.start()
    #
    #     return self

    def run(self):
        # keep looping infinitely until the thread is stopped
        while not self.stop_flag.is_set():
            self.ret, self.frame = self.cam.read()

    def getFrame(self):
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.cam.release()
        self.stop_flag.set()
