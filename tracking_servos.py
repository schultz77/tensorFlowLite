from servo import Servo

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