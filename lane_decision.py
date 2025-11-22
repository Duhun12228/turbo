#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
import math
import time


class LaneDecision:
    def __init__(self):
        rospy.init_node('lane_decision')
        self.cv_bridge = CvBridge()

        # --- ROS I/O ---
        self.image_sub = rospy.Subscriber(
            '/usb_cam/image_rect_color/compressed', CompressedImage, self.image_cb, queue_size=1
        )
        self.binary_img_sub = rospy.Subscriber('/binary_img',Image,self.binary_img_cb,queue_size=10)
        self.mission_sub = rospy.Subscriber(
            '/mission_num', Float64, self.mission_cb, queue_size=1)
        self.mission_pub = rospy.Publisher('/mission_num', Float64, queue_size=10)
        self.ack_pub_1 = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=10)
        self.decision_sub = rospy.Subscriber('/lane_decision',Float64,self.decision_cb,queue_size=10)
        
        
        self.bgr = None
        self.binary_img = None
        self.current_mission = None
        self.decision = 'right'
        self.state = None

    def image_cb(self,msg):
        self.bgr = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
    
    def binary_img_cb(self,msg):
        self.binary_img = self.cv_bridge.imgmsg_to_cv2(msg)
    
    def mission_cb(self,msg):
        pass

    def decision_cb(self,msg):
        num = msg.data

        if num == 1: #if 1 => right:
            self.decision = 'right'
        elif num == 2:
            self.decision = 'left'
        else:
            pass

    def sliding_window_right(self,img,n_windows=10,margin = 12,minpix = 5):
        y = img.shape[0]
        x = img.shape[1]
        h, w = img.shape[:2]
        masked = img.copy()
        masked[:, :w//2] = 0
        img = masked  
        
        histogram = np.sum(img[y//2:,:], axis=0)   
        midpoint = int(histogram.shape[0]/2)
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint
        
        window_height = int(y/n_windows)
        nz = img.nonzero()

        left_lane_inds = []
        right_lane_inds = []
    
        lx, ly, rx, ry = [], [], [], []

        out_img = np.dstack((img,img,img))*255

        for window in range(n_windows):
                
            win_yl = y - (window+1)*window_height
            win_yh = y - window*window_height


            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            cv.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2) 

            # 슬라이딩 윈도우 박스(녹색박스) 하나 안에 있는 흰색 픽셀의 x좌표를 모두 모은다.
            good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]

            right_lane_inds.append(good_right_inds)

            # 구한 x좌표 리스트에서 흰색점이 5개 이상인 경우에 한해 x 좌표의 평균값을 구함. -> 이 값을 슬라이딩 윈도우의 중심점으로 사용
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nz[1][good_right_inds]))

            rx.append(rightx_current)
            ry.append((win_yl + win_yh)/2)

        right_lane_inds = np.concatenate(right_lane_inds)

        rfit = np.polyfit(np.array(ry),np.array(rx),1)
        
        out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]
        cv.imshow("right_viewer", out_img)

        return rfit

    def cal_center_line(self, rfit):
        """
        lfit, rfit : np.polyfit으로 구한 왼쪽/오른쪽 차선의 1차 다항식 계수
                     x = a*y + b 형태 (len == 2)
        반환값:
            yaw   : 중앙 차선의 진행 방향 각도 (라디안)
            error : 차량(이미지 중앙) 기준, 차선 중앙의 x 오프셋(px)
        """
        a,b = rfit
        # 1) 왼쪽/오른쪽 차선의 계수를 평균내서 '중앙선' 계수로 사용
        cfit = [a,b-120] # [a, b]

        h, w = 170, 640

        y_eval = h * 0.75  # 이미지 높이의 3/4 지점

        # 3) 해당 y에서 중앙선의 x 좌표 계산
        a, b = cfit
        x_center = a * (y_eval) + b 

        # 4) 중앙선의 기울기(dx/dy) 계산 후, yaw 각도(라디안) 계산
        #    x = a*y^2 + b*y + c  → dx/dy = 2*a*y + b
        dx_dy = a
        yaw = np.arctan(dx_dy)  # 전방(y 방향) 기준 x의 변화량에 대한 각도

        # 5) 차량을 이미지 가로 중앙에 있다고 가정하고, 중앙선과의 오프셋 계산
        img_center_x = w / 2.0
        error =  - x_center + img_center_x  # >0: 중앙선이 오른쪽에 있음, <0: 왼쪽에 있음

        return yaw, error
    
    def cal_steering(self,yaw,error,gear=3,k=0.005,yaw_k=1.0): #각도들은 라디안, 거리는 px값, 속도는 0~1사이 스케일값 m/s
        if gear == 3: #기본 주행
            base_speed = 0.30
        elif gear == 2: # 가속 주행
            base_speed = 0.55
        elif gear ==1: # 감속 주행
            base_speed = 0.25
        
        steering = yaw_k*yaw + np.arctan2(k*error,base_speed)
        self.steer = steering

        speed = base_speed #steering에 따른 속도 조절이 필요?

        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'lane_decision'
        msg.drive.steering_angle = steering
        msg.drive.speed = speed

        self.ack_pub_1.publish(msg)

    def follow_right(self,img):
        rfit = self.sliding_window_right(img)
        yaw,error = self.cal_center_line(rfit)
        self.cal_steering(yaw,error)

    def follow_left(self):
        pass

    def main(self):
        if self.binary_img is None:
            return
            
        if self.state == None:
            if self.decision is None:
                print('Sign not detected!')
            
            elif self.decision == 'right':
                print('lane is right!!!')
                self.state = 'following right lane'
                self.follow_right(self.binary_img)

        elif self.state == 'following right lane':
            self.follow_right(self.binary_img)



if __name__ == '__main__':
    try:
        ld = LaneDecision()
        rate = rospy.Rate(25)  # 30Hz

        while not rospy.is_shutdown():
            ld.main()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass