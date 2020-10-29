#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import h5py
import termios
import sys
import tty
import atexit
from select import select
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import tensorflow as tf
import agent
import RPi.GPIO as GPIO
import threading

# for sonic Pin setting
GPIO.setmode(GPIO.BCM)
trig = 3
echo = 2
GPIO.setwarnings(False)
GPIO.setup(trig, GPIO.OUT)
GPIO.setup(echo, GPIO.IN)
is_obstacle = False

# for drive mode - state machine
DRIVE = 1
AVOID = 2
AVOID_DRIVE = 3
MODE = DRIVE

# for distance threshold
THRESH_SONIC = 2                                               
THRESH_LDS = .35

# for LDS value save
front_left = []
front_right = []
left_left = []
left_right = []


# for Key Input
class KBHit:
    def __init__(self):
        '''Creates a KBHit object that you can call to do various keyboard things.
        '''

        # Save the terminal settings
        self.fd = sys.stdin.fileno()
        self.new_term = termios.tcgetattr(self.fd)
        self.old_term = termios.tcgetattr(self.fd)
        
        # New terminal setting unbuffered
        self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

        # Support normal-terminal reset at exit
        atexit.register(self.set_normal_term)


    def set_normal_term(self):
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)


    def getch(self):
        return sys.stdin.read(1)


    def kbhit(self):
        ''' Returns True if keyboard character was hit, False otherwise.
        '''

        dr,dw,de = select([sys.stdin], [], [], 0)
        return dr != []
        
# for Ultra Sonic
def ultrasonic():
    '''return : distance
    '''
    GPIO.output(trig, 0)
    time.sleep(0.00001)
    pulse_start = 0
    pulse_end = 0.0235
    start = time.time()
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)
    
    while GPIO.input(echo) == 0 and time.time() - start <= 0.0235:
        pulse_start = time.time()
        
    while GPIO.input(echo) == 1 and time.time() - start <= 0.0235:
        pulse_end = time.time()
        
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17000
    distance = round(distance, 2)

    return distance

# for LDS
def scaning(msg):
    '''save lidar sensor values
        run through sub thread
    '''
    global MODE, front_right, front_left, left_left, left_right
    front_left = [float('{0:0.6f}'.format(i)) if i != 0.0 else 1 for i in msg.ranges[:10]]    
    front_right = [float('{0:0.6f}'.format(i)) if i != 0.0 else 1 for i in msg.ranges[350:359]]
        
    left_right = [float('{0:0.6f}'.format(i)) if i != 0.0 else 1 for i in msg.ranges[80:90]]
    left_left = [float('{0:0.6f}'.format(i)) if i != 0.0 else 1 for i in msg.ranges[90:100]]
        
    if(float(min(front_left)) <= THRESH_LDS or float(min(front_right)) <= THRESH_LDS):
        if(MODE == DRIVE):
            MODE = AVOID
            print('[debug] - avoid mode!')
    
def lds():
    '''ROS node
    '''
    sub = rospy.Subscriber('/scan', LaserScan, scaning)


# for Key Input - deprecated! not use
def getkey():
    fd = sys.stdin.fileno()
    original_attributes = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_attributes)
    return ch


# for Drive, collect
class Collector:
    ''' core class
        this class include drive, collect, control, save dataset.. etc
    '''
    def __init__(self):
        self.observations = []
        self.labels = []
        rospy.init_node('collector_node')
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.move = Twist()
        self.speed = .2
        self.angular = .3
    
    # control
    def move_forward(self):
        self.move.linear.x = self.speed
        self.move.angular.z = 0
        self.pub.publish(self.move)
        command = [0,1,0,0]
        return command
        
    def turn_right(self):
        self.move.linear.x = .1
        self.move.angular.z = -self.angular
        self.pub.publish(self.move)
        command = [0,0,1,0]
        return command
        
    def turn_left(self):
        self.move.linear.x = .1
        self.move.angular.z = self.angular
        self.pub.publish(self.move)
        command = [1,0,0,0]
        return command
    
    def no_line(self):
        if MODE == AVOID_DRIVE :
            self.move.linear.x = self.speed/1.5
            self.move.angular.z = -self.angular/2
            print("AVOID_NOLINE")
        elif MODE == DRIVE :
            self.move.linear.x = -self.speed/2
            self.move.angular.z = 0
        
        self.pub.publish(self.move)
        command = [0,0,0,1]
        return command
    
    def turn_90(self, direc, delay=5.5):            
        self.move.angular.z = -self.angular if direc == 'r' else self.angular
        self.move.linear.x = 0
        self.pub.publish(self.move)
        time.sleep(delay)
        
        self.move.angular.z = 0
        self.pub.publish(self.move)
        
    # control
    def stop_turtlebot(self):
        self.move.linear.x = 0
        self.move.angular.z = 0
        self.pub.publish(self.move)
    
    def control(self):
        # just only control if exist
        key = getkey()
        if(key == 'w'):
            command = self.move_forward()
            return command
        elif(key == 'd'):
            command = self.turn_right()
            return command
        elif(key == 'a'):
            command = self.turn_left()
            return command
        #JH
        elif(key == 'e'):
            command = no_line()
            return command
        else:
            return 0    
        
    def adjust(self) :
        ''' when turtlebot is in avoid mode, adjust direcion itself
        '''
        LR = sum(left_right[5:])/5
        LL = sum(left_left[:5])/5
        if (LR > LL) :
            while True :
                LR = sum(left_right[5:])/5
                LL = sum(left_left[:5])/5
                
                time.sleep(0.05)
                print(LR,LL , LR-LL)
                #if ( LR - LL > 0.00002 ) :
                print('[debug] - adjust(left)!')
                self.move.linear.x = 0
                self.move.angular.z = self.angular
                self.pub.publish(self.move)
                    
                if (LR-LL < 0 or LR-LL == 0):
                    break
                
        elif (LL > LR) :
            while True :
                LR = sum(left_right[5:])/5
                LL = sum(left_left[:5])/5
                
                time.sleep(0.05)
                print(LR,LL , LL-LR)
                
                #if ( LL - LR > 0.00002 ) :
                print('[debug] - adjust(right)!')
                self.move.linear.x = 0
                self.move.angular.z = -self.angular
                self.pub.publish(self.move)
                    
                if (LL-LR < 0 or LL-LR == 0):
                    break
                #else:
                 #   break
            
    def steer_lds(self, d1, d2):
        ''' when turtlebot is in avoid mode, move with lidar sensor
        '''
        global MODE
        first_d = d1
        # forward
        self.move.linear.x = .1
        self.move.angular.z = 0
        self.pub.publish(self.move)
        time.sleep(first_d)
        
        # check left lds
        while(True):
            if(float(min(left_left)) <= THRESH_LDS or float(min(left_right)) <= THRESH_LDS):
                # move forward a little bit
                self.move.linear.x = .1
                self.move.angular.z = 0
                self.pub.publish(self.move)
                time.sleep(.5)
                first_d += .5
            else:
                # move forward a little bit
                print('[debug] - no obstacle, turn', min(left_left))
                self.move.linear.x = .1
                self.move.angular.z = 0
                self.pub.publish(self.move)
                time.sleep(1.0)
                self.turn_90('l')
                # move forward a little bit
                self.move.linear.x = .1
                self.move.angular.z = 0
                self.pub.publish(self.move)
                time.sleep(3.5)
                break
        
        # check left lds
        while(True):
            if(float(min(left_left)) <= THRESH_LDS * 1.5 or float(min(left_right)) <= THRESH_LDS * 1.5):
                # move forward a little bit
                self.move.linear.x = .1
                self.move.angular.z = 0
                self.pub.publish(self.move)
                time.sleep(.5)
            else:
                print('[debug] - no obstacle, turn', min(left_left))
                self.move.linear.x = .1
                self.move.angular.z = 0
                self.pub.publish(self.move)
                time.sleep(.5)
                self.turn_90('l',d2)
                break
        
        self.move.linear.x = .1
        self.move.angular.z = 0
        self.pub.publish(self.move)
        
        MODE = AVOID_DRIVE
        print("steerFunction END! MODE = ",MODE)

    
    def control_use_model(self, action):
        ''' CNN's output is injected
            and send command to turtlebot
        '''
        global MODE
        if(action == 0):
            self.turn_left()
            MODE = DRIVE
        elif(action == 1):
            self.move_forward()
            MODE = DRIVE
        elif(action == 2):
            self.turn_right()
            MODE = DRIVE
        elif(action == 3):
            self.no_line()
        
    def collect(self):
        ''' collect dataset through manual keyboard control 
        '''
        print('> start collecting...')
        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 20
        rawCapture = PiRGBArray(camera, size=(640, 480))
        time.sleep(.5)
        
        cnt = 0
        #JH
        command = [0, 1, 0, 0]
        start = time.time()
        
        for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            cnt += 1
            frame = f.array
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_bin, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            binary = gray[160:, :]
            #cv2.imshow('test', frame)
            #key = cv2.waitKey(1) & 0xFF
            
            key = getkey()

            if(key == 'w'):
                command = self.move_forward()
            elif(key == 'd'):
                command = self.turn_right()
            elif(key == 'a'):
                command = self.turn_left()
            
            #JH
            elif(key == 'e'):
                command = self.no_line()
            
            elif(key == 'q'):
                print('> [Exit_code] - stop recording, stop turtlebot')
                self.stop_turtlebot()
                break
            
            self.observations.append(binary) # add input data
            self.labels.append(command) # add label data
            rawCapture.truncate(0)

        camera.close()
        cv2.destroyAllWindows()        
                      
                      
    def drive(self):
        ''' autonomous driving through trained weight
            can interfere by pressing 'w','a','d' and control
            can return to auto drive mode by pressing 'q'
        '''
        print('> start drive')
        global MODE
        
        with tf.Session() as sess:
            kb = KBHit()
            model = agent.CNN(320, 640, sess)
            saver = tf.train.Saver(tf.trainable_variables())
            print('> turtlebot drive now')
            save_path = '/home/pi/Downloads/model'
            saver.restore(sess, save_path)
            print('> model parameter loded : {}'.format(save_path))
            camera = PiCamera()
            camera.resolution = (640, 480)
            camera.framerate = 32
            rawCapture = PiRGBArray(camera, size=(640, 480))
            time.sleep(.2)
            print('> running...')
            cnt = 0
            #JH
            command = [0, 1, 0 ,0]
            
            base = time.time()
            global is_obstacle
            is_interfere = False
            
            is_avoid_start = True
            t = threading.Thread(target=lds)
            t.daemon = True
            t.start()
            #rospy.spin()
            
            for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                print("MODE",MODE)
                ## drive mode
                if(MODE == DRIVE or MODE == AVOID_DRIVE):
                    # check ultrasonic
                    if(time.time() - base >= .5):
                        dist = ultrasonic()
                        if(dist <= THRESH_SONIC):
                            #print('[debug] - check ultrasonic', dist)
                            is_obstacle = True
                            self.stop_turtlebot()
                        else:
                            is_obstacle = False
                        base = time.time()
                        
                    if(is_obstacle):
                        rawCapture.truncate(0)
                        continue
                        
                    # check camera
                    frame = f.array
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = gray[160:, :]
                    gray = cv2.medianBlur(gray, 21)
                    rawCapture.truncate(0)
                    
                    # check keyboardHit
                    if kb.kbhit():
                        c = kb.getch()
                        is_interfere = True
                        
                        if(c == 'q'):
                            self.stop_turtlebot()
                            break
                        elif(c == 's'):
                            is_interfere = False
                            continue
                        elif(c == 'w'):
                            command = self.move_forward()
                        elif(c == 'a'):
                            command = self.turn_left()
                        elif(c == 'd'):
                            command = self.turn_right()
                        
                        #JH
                        elif(c == 'e'):
                            command = self.no_line()
                            
                        #print('[debug] - keyhit', command)
                        self.observations.append(gray) # add input data
                        self.labels.append(command) # add label data
                    elif(not is_interfere):
                        policy = model.policy(gray.reshape(1,320,640,1))
                        #print('[debug] policy shape : {}'.format(policy))
                        action = np.argmax(policy) # CNN --> policy --> action
                        self.control_use_model(action)
                    
                ## avoid mode
                elif(MODE == AVOID):
                    global front_left, front_right, left_left, left_right
                    if(is_avoid_start):
                        #print('[debug] - turn 90 start!')
                        self.turn_90('r')
                        is_avoid_start = False
                        #print('[debug] - turn 90 end!')
                        self.adjust()
                        print('[debug] - adjust end!')
                    
                    # general avoid
                    self.steer_lds(2.5, 4.5)

                    rawCapture.truncate(0)
                    #MODE = DRIVE
                    is_avoid_start = True
                    #print('[debug] - back to drive mode!')
                    print("AVOID_IF END / MODE = ",MODE)
                    
                  
            cv2.destroyAllWindows()
            camera.close()
            kb.set_normal_term()
                                        
            
    def make_data(self, file_name='dataset'):
        ''' save dataset file for 'h5py' file format
        '''
        # make initial dataset file and stack
        file = file_name + '.h5'
        print('> create database file - {}'.format(file))
        with h5py.File(file, 'w') as f:
            f.create_dataset('observation', data=self.observations, dtype='int32')
            f.create_dataset('label', data=self.labels, dtype='int32')
            
            
    def add_data(self):
        # aggregation
        print('> add recorded data')
        with h5py.File('dataset.h5', 'a') as f:
            obs = np.array(f['observation'])
            labs = np.array(f['label'])
            new_obs = np.array(self.observations)
            new_labs = np.array(self.labels)
            agg_obs = list(obs) + list(new_obs)
            agg_labs = list(labs) + list(new_labs)
            del(f['observation'])
            del(f['label'])
            f['observation'] = agg_obs
            f['label'] = agg_labs
            
    def read_data(self, idx):
        with h5py.File('dataset.h5', 'r') as f:
            arr = np.array(f['observation'])
            arr_label = np.array(f['label'])
            idx = len(arr) - 1 if idx >= len(arr) else idx
            print('> [debug] - {} th data \n observatin : {} \n label : {}'.format(idx, arr[idx], arr_label[idx]))


##
if(__name__ == '__main__'):
    start = time.time()
    collector = Collector()
    #collector.collect()
    collector.drive()
    rospy.spin()
    collector.make_data('dataset_1')
