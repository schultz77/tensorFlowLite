import cv2
import time
from picamera2 import Picamera2
from threading import Thread
from servo import Servo

from tflite_support.task import core, processor, vision

import utils

class Tracking:

    __slots__ = "panAngle",'tiltAngle', 'motorPan','motorTilt','MAX_ANGLE','MIN_PIX_ERR','PIX2DEG_RATE','disp','filtering_box','panAngle_pre','tiltAngle_pre','cog_contour','low_freq_factor', 'x_motor', 'y_motor', 'error_x', 'error_y'


    def __init__(self, 
                 panAngle=0, 
                 tiltAngle=0, 
                 motorPan=13, 
                 motorTilt=12,
                 MAX_ANGLE={'PAN': 90, 'TILT': 40}, 
                 MIN_PIX_ERR=35, 
                 PIX2DEG_RATE=70,
                 disp={'width': 1280, 'height':720},
                 filtering_box = True,
                 panAngle_pre = 0,
                 tiltAngle_pre = 0,
                 cog_contour = {'x': 0, 'y': 0},
                 low_freq_factor = 0.8,
                 x_motor= None, y_motor= None, error_x = 0, error_y = 0) -> None:
         
         self.x_motor = Servo(motorPan)
         self.y_motor = Servo(motorTilt)
         self.panAngle = panAngle
         self.tiltAngle = tiltAngle
         self.MAX_ANGLE = MAX_ANGLE
         self.MIN_PIX_ERR = MIN_PIX_ERR
         self.PIX2DEG_RATE = PIX2DEG_RATE
         self.cog_contour = cog_contour
         self.low_freq_factor = low_freq_factor
         self.disp = disp
         self.filtering_box = filtering_box
         self.panAngle_pre = panAngle_pre
         self.tiltAngle_pre = tiltAngle_pre
         self.error_x = error_x
         self.error_y = error_y

        #  self.error_x_prev = 0
        #  self.error_x_sum = 0
         

         self.x_motor.set_angle(self.panAngle)
         self.y_motor.set_angle(self.tiltAngle)

    

    def low_pass_filter(self, dim={'x_box':0,'y_box':0,'width_box': 0, 'height_box': 0}):
                 
        if self.cog_contour['x'] and self.filtering_box:
           self.cog_contour['x'] = self.low_freq_factor * self.cog_contour['x'] + (1-self.low_freq_factor) * (dim['x_box']+dim['width_box']/2)
        else:
           self.cog_contour['x'] = dim['x_box']+dim['width_box']/2
        
        if self.cog_contour['y'] and self.filtering_box:
            self.cog_contour['y'] = self.low_freq_factor * self.cog_contour['y'] + (1-self.low_freq_factor) * (dim['y_box']+dim['height_box']/2)
        else:
            self.cog_contour['y'] = dim['y_box']+dim['height_box']/2
        

    # def start(self):
    #     # start the thread for moving the motors
    #     self.mov_thread=Thread(target=self.movement, args=())
    #     self.mov_thread.daemon=True
    #     self.mov_thread.start()

    
    def movement(self, dim={'x_box':0,'y_box':0,'width_box': 0, 'height_box': 0}):
    
        
        self.low_pass_filter(dim)

        # calculating distance between cog_contour and the center of the window - PID control
        KP_x = 1/self.PIX2DEG_RATE
        KD_x = KP_x/2
        KI_x = KD_x/2

        # calculating distance between cog_contour and the center of the window
        self.error_x = -(self.cog_contour['x'] - self.disp['width']/2)
        self.panAngle += self.error_x/self.PIX2DEG_RATE
        
        # PID control
        # self.panAngle += (self.error_x*KP_x) + (self.error_x_prev*KD_x) + (self.error_x_sum*KI_x)
        # self.error_x_prev = self.error_x
        # self.error_x_sum += self.error_x

        # print('cog_contour[x]: {} KP: {}'.format(self.cog_contour['x'], self.error_x/self.PIX2DEG_RATE ))
        # print('error_x: {} panAngle: {} pan_angle_pre {}'.format(self.error_x, self.panAngle, self.panAngle_pre ))
        
        if self.panAngle < -self.MAX_ANGLE['PAN']:
            self.panAngle = -self.MAX_ANGLE['PAN']
        if self.panAngle > self.MAX_ANGLE['PAN']:
            self.panAngle = self.MAX_ANGLE['PAN']
        
        if abs(self.error_x) > self.MIN_PIX_ERR and abs(abs(self.panAngle) - abs(self.panAngle_pre)) > 5:
            # print('error_x_in:: {} panAngle_cmd: {}'.format(self.error_x, self.panAngle ))
            self.x_motor.set_angle(self.panAngle) # (+) conterclockwise --> right / (-) clockwise --> left
            self.panAngle_pre = self.panAngle
            
        # print('panAngle_actual: ', self.x_motor.get_angle())

        self.error_y = self.cog_contour['y'] - self.disp['height']/2
        self.tiltAngle += self.error_y/self.PIX2DEG_RATE
                                        
        if self.tiltAngle > self.MAX_ANGLE['TILT']:
            self.tiltAngle = self.MAX_ANGLE['TILT']
        if self.tiltAngle < -self.MAX_ANGLE['TILT']:
            self.tiltAngle=-self.MAX_ANGLE['TILT']
                        
        if abs(self.error_y) > self.MIN_PIX_ERR and abs(abs(self.tiltAngle) - abs(self.tiltAngle_pre)) > 5:
            self.y_motor.set_angle(self.tiltAngle) # (+) down / (-) up
            self.tiltAngle_pre = self.tiltAngle
            


class PiVideoStream:
	
    def __init__(self, resolution=(1280, 720), framerate=30, webCam_Flag=False):
                
                self.model='efficientdet_lite0.tflite'
                self.num_threads = 4
                
                self.frame = []
                
                # initialize the camera and stream     
                self.picam2 = Picamera2()
                self.picam2.preview_configuration.main.size = resolution
                self.picam2.preview_configuration.main.format = "RGB888"
                self.picam2.preview_configuration.controls.FrameRate=framerate
                self.picam2.preview_configuration.align()
                self.picam2.configure("preview")
                self.picam2.start()
                
                # checking webcam --> bash: v4l2-ctl --list-devices
                self.webCam='/dev/video2'
                self.cam=cv2.VideoCapture(self.webCam)
                # self.cam.release()
                self.webCam_Flag = webCam_Flag
                self.ret = None
                
                self.base_options=core.BaseOptions(file_name=self.model,use_coral=False,num_threads=self.num_threads)
                self.detection_options=processor.DetectionOptions(max_results=8, score_threshold=.3)
                self.options=vision.ObjectDetectorOptions(base_options=self.base_options,detection_options=self.detection_options)
                self.detector=vision.ObjectDetector.create_from_options(self.options)
            
                # if the thread should be stopped
                self.stopped = False
		
    def start(self):
        # start the thread to read frames from the video stream
        self.thread=Thread(target=self.update, args=())
        self.thread.daemon=True
        self.thread.start()

        return self
    
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            if self.webCam_Flag:
                self.ret, self.frame = self.cam.read()
                
            else:
                self.frame=self.picam2.capture_array()
            
            if self.stopped:
                self.cam.release()
                break
            
    def getFrame(self):
        return self.frame
    

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True



# model='efficientdet_lite0.tflite'
# num_threads = 4



#picam2=Picamera2()
#picam2.preview_configuration.main.size=(dispW,dispH)
#picam2.preview_configuration.main.format='RGB888'
#picam2.preview_configuration.align()
#picam2.configure("preview")

#picam2.start()

# checking webcam --> bash: v4l2-ctl --list-devices
# webCam='/dev/video2'

# cam=cv2.VideoCapture(webCam)
# cam.release()
#cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(640))
#cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(480))
#cam.set(cv2.CAP_PROP_FPS,15)


dispW=int(1280)
dispH=int(720)
frameRate = 30
fps=0
textPos=(30,60)
font=cv2.FONT_HERSHEY_SIMPLEX
textHeight=1.5
textWeight=3
textColor=(0,0,180)

boxColor = (255,0,0)
boxWeight = 2

labelHeight = 1
labelColor = (0, 255, 0)
labelWeight= 2


webCam_Flag= False
#base_options=core.BaseOptions(file_name=model,use_coral=False,num_threads=num_threads)
#detection_options=processor.DetectionOptions(max_results=8, score_threshold=.3)
#options=vision.ObjectDetectorOptions(base_options=base_options,detection_options=detection_options)
#detector=vision.ObjectDetector.create_from_options(options)

# instatiating the class objects 
myCam = PiVideoStream(resolution=(dispW,dispH), framerate=frameRate, webCam_Flag=webCam_Flag)

# starting the thread for capturing frames
myCam.start()

# tStart=time.time()

while True:
    frame= myCam.getFrame()
    tStart=time.time()
    
    if len(frame):
        if not webCam_Flag:
            frame=cv2.flip(frame,-1) # flipping frame by 180 degrees
        # ret, im = cam.read()
        # im = picam2.capture_array()
        # im=cv2.flip(im,-1)
        imRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        imTensor=vision.TensorImage.create_from_array(imRGB)
        myDetects=myCam.detector.detect(imTensor)
        #print('\n', myDetects)
        for myDetect in myDetects.detections:
            UL = (myDetect.bounding_box.origin_x, myDetect.bounding_box.origin_y)
            LR = (UL[0] + myDetect.bounding_box.width, UL[1] + myDetect.bounding_box.height)
            frame = cv2.rectangle(frame,UL, LR,boxColor, boxWeight)
            objName = myDetect.categories[0].category_name
            cv2.putText(frame,objName,UL, font, labelHeight, labelColor, labelWeight)
            

        # image=utils.visualize(frame,myDetects)
        
        cv2.putText(frame,str(int(fps))+' FPS',textPos,font,textHeight,textColor,textWeight)
        cv2.imshow('Camera',frame)
        if cv2.waitKey(1)==ord('q'):
            myCam.stop()
            break
        
        tEnd=time.time()
        loopTime = tEnd - tStart
        fps = 0.8*fps + 1/loopTime
    

cv2.destroyAllWindows()
