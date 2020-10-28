#!/usr/bin/env python
"""
Created on Fri Nov  9 10:25:22 2018
@author: mason
"""
'''libraries'''
import time
import numpy as np
import rospy
import roslib
import cv2

from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler

global LSD
global cap
LSD = cv2.createLineSegmentDetector(0)  #이미지에서 사각형 선을 인식하는 알고리즘
cap = cv2.VideoCapture(0)
''' class '''
class robot():
    def __init__(self):
        rospy.init_node('robot_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1) #Twist 메세지 타입을 사용하는 'cmd_vel'토픽에게 노드를 발행

    def keeping(self,hsv):
        global LSD
        vel_msg=Twist()   # linear, angular 사용하기 위한 선언
        crop_L=hsv[350:410,40:220]
        crop_R=hsv[350:410,402:582]
        L_mask = cv2.inRange(crop_L,(21,50,100),(36,255,255)) # inRange(hsv,Low,High) : 추출하고자 하는 색이 Low~High범위 안이면 흰색 return 범위 밖이면 검은색으로 return
        L2_mask = cv2.inRange(crop_L,(36,0,165),(255,255,255))
        R_mask = cv2.inRange(crop_R,(36,0,165),(255,255,255))
        R2_mask = cv2.inRange(crop_R,(21,50,100),(36,255,255))
      
        yello_line = LSD.detect(L_mask)   # .detect() : input값에서 line을 찾아준다
        yello_line2 = LSD.detect(L2_mask)
        white_line = LSD.detect(R_mas
        white_line2 = LSD.detect(R2_mask)
        if yello_line[0] is None and yello_line2[0] is None:  # 노란색선이 인식되지 않으면 왼쪽회전
            vel_msg.linear.x = 0.05   # linear.x : 직선속도
            vel_msg.angular.z = 0.6   # angular.z : 회전속도
        elif white_line[0] is None and white_line2[0] is None:   # 하얀색선이 인식되지 않으면 오른쪽회전
            vel_msg.linear.x = 0.05
            vel_msg.angular.z = -0.6
        else :    # 그 외 경우 = 직진
            vel_msg.linear.x = 0.2
            vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg) # cmd_vel에 vel_msg값을 전달

    def imageupdate(self):
        global cap
        ret, image=cap.read() # image값에 읽은 프레임이 저장, ret에는 프레임읽기에 성공하면 True return
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
