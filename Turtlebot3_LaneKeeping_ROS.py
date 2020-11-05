#!/usr/bin/env python
'''libraries'''
import time
import numpy as np
import rospy
import roslib
import cv2

from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from tf.transformations import euler_from_quaternion, quaternion_from_euler

global LSD
LSD = cv2.createLineSegmentDetector(0)

''' class '''
class robot():
    def __init__(self):
        rospy.init_node('robot_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.img_subscriber = rospy.Subscriber('/raspicam_node/image/compressed',CompressedImage,self.callback_img)

    def callback_img(self,data):
        np_arr = np.fromstring(data.data, np.uint8) # string data --> np data
        self.image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        
    def keeping(self,hsv):
        global LSD
        vel_msg=Twist()
        crop_L=hsv[420:480,120:240]
        crop_R=hsv[420:480,400:500]
        L_mask = cv2.inRange(crop_L,(21,50,100),(36,255,255)) # Yellow lane
        R_mask = cv2.inRange(crop_R,(40,0,180),(115,30,255)) # White lane
      
        yello_line = LSD.detect(L_mask)
        white_line = LSD.detect(R_mask)

        if yello_line[0] is None :
            vel_msg.linear.x = 0.03
            vel_msg.angular.z = 0.35
        elif white_line[0] is None :
            vel_msg.linear.x = 0.03
            vel_msg.angular.z = -0.35
        else :
            vel_msg.linear.x = 0.08
            vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)

    def imageupdate(self):
        image=self.image_np
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        return image,hsv
        
turtle=robot()
time.sleep(1.2)
if __name__=='__main__':
    while 1:
        try:
            img,hsv=turtle.imageupdate()
            turtle.keeping(hsv) 
        except :
           print('got error')
