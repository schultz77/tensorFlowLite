import cv2
import time
from tracking_servos import Tracking
from tflite_support.task import vision
from Frame_Capture_tfl_thrd_stop import piVideoStream
# import utils

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

# instatiating the class objects 
myCam = piVideoStream(resolution=(dispW,dispH), framerate=frameRate)
# starting the thread for capturing frames
myCam.start()

myTrack = Tracking(tiltAngle=-30, disp={'width': dispW, 'height':dispH}, filtering_box=True, low_freq_factor=.9)

while True:
    frame= myCam.getFrame()
    tStart=time.time()
    
    if len(frame):
        frame=cv2.flip(frame,-1) # flipping frame by 180 degrees
        imRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        imTensor=vision.TensorImage.create_from_array(imRGB)
        myDetects=myCam.detector.detect(imTensor)
        #print('\n', myDetects)
        x_box, y_box = 0, 0
        box_width, box_height = 0, 0
        for myDetect in myDetects.detections:
            objName = myDetect.categories[0].category_name
            if objName == "cell phone":
                x_box, y_box = myDetect.bounding_box.origin_x, myDetect.bounding_box.origin_y
                box_width, box_height = x_box + myDetect.bounding_box.width, y_box + myDetect.bounding_box.height
                # UL = (x_box, y_box)
                # LR = (box_width, box_height)

                # frame = cv2.rectangle(frame,UL, LR,boxColor, boxWeight)
                # objName = myDetect.categories[0].category_name
                # cv2.putText(frame,objName,UL, font, labelHeight, labelColor, labelWeight)
                # break

            UL = (myDetect.bounding_box.origin_x, myDetect.bounding_box.origin_y)
            LR = (UL[0] + myDetect.bounding_box.width, UL[1] + myDetect.bounding_box.height)
            frame = cv2.rectangle(frame,UL, LR,boxColor, boxWeight)
            objName = myDetect.categories[0].category_name
            cv2.putText(frame,objName,UL, font, labelHeight, labelColor, labelWeight)

        if x_box:
            dim={'x_box': x_box,'y_box':y_box,'width_box':int(box_width/2), 'height_box': int(box_height/2)}
            myTrack.movement(dim)
            
        # image=utils.visualize(frame,myDetects)
        
        cv2.putText(frame,str(int(fps))+' FPS',textPos,font,textHeight,textColor,textWeight)
        cv2.imshow('Camera',frame)
        if cv2.waitKey(1)==ord('q'):
            myCam.stop()  # When everything done, release the capture
            myCam.join()
            break
        
        tEnd=time.time()
        loopTime = tEnd - tStart
        fps = 0.8*fps + 1/loopTime
    

cv2.destroyAllWindows()
