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


class LaneFollow:
    def __init__(self):
        rospy.init_node('lane_follow')
        self.cv_bridge = CvBridge()
        
        # --- ROS I/O ---
        self.image_sub = rospy.Subscriber(
            '/usb_cam/image_rect_color/compressed', CompressedImage, self.image_cb, queue_size=1
        )

        self.mission_sub = rospy.Subscriber(
            '/mission_num', Float64, self.mission_cb, queue_size=1)
        self.mission_pub = rospy.Publisher('/mission_num', Float64, queue_size=10)

        self.ack_pub = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=10)
        self.ack_pub_1 = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=10)
        
        self.roi_img_pub = rospy.Publisher('/roi_img', Image, queue_size=10)
        self.binary_img_pub = rospy.Publisher('/binary_img',Image,queue_size = 10)

        self.debug_publisher1 = rospy.Publisher('/debugging_image1', Image, queue_size=10)
        self.debug_publisher2 = rospy.Publisher('/debugging_image2', Image, queue_size=10)

        # --- 파라미터 ---
        self.img_width  = rospy.get_param('~img_width', 640)
        self.img_height = rospy.get_param('~img_height', 480)

        # HSV (화이트) 기본값: 상황에 맞게 파라미터로 조정 가능
        self.white_lower = np.array(rospy.get_param('~white_lower', [0, 0, 180]), dtype=np.uint8)
        self.white_upper = np.array(rospy.get_param('~white_upper', [180, 40, 255]), dtype=np.uint8)

        self.start_mission = 0

        self.src_points= np.float32([
            [0, 310],
            [640, 310],
            [0, 480],
            [640, 480]
        ])
        self.dst_points= np.float32([
            [0,   310],
            [640,   310],
            [225 , 480],
            [415, 480]
        ])

        self.warp_mat = cv.getPerspectiveTransform(self.src_points,self.dst_points)
        self.inv_warp_mat = cv.getPerspectiveTransform(self.dst_points,self.src_points)
        
    
        self.bgr = None
        self.warp_img = None
        self.white_img = None
        self.filtered_img = None
        self.gaussian_sigma = 1
        self.gear = 3 # 3.이 default
        self.yaw = 0
        self.error = 0
        self.steer = 0
        
        self.zone = None
        

    def image_cb(self, image_msg):
        self.bgr = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg)
    def mission_cb(self,msg):
        pass
    def warpping(self,img):
        h,w = img.shape[:2]
        warp_img = cv.warpPerspective(img,self.warp_mat,(w,h))
        return warp_img
    
    def Gaussian_filter(self,img):
        filtered_img = cv.GaussianBlur(img,(0,0),self.gaussian_sigma)
        return filtered_img
    
    def white_color_filter_hsv(self,img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        white_hsv = cv.inRange(hsv,self.white_lower,self.white_upper)
        masked_img = cv.bitwise_and(img,img,mask=white_hsv)
        return masked_img
    
    #반사 등 제거 후 명확하게
    def binary_filter(self,img):
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        _,binary =  cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
        return binary

    def roi_set(self,img):
        roi_img = img[310:480,0:640]
        return roi_img
    
    def cal_steering(self,yaw,error,gear=3,k=0.005,yaw_k=1.0): #각도들은 라디안, 거리는 px값, 속도는 0~1사이 스케일값 m/s
        gear = self.gear

        if gear == 3: #기본 주행
            base_speed = 0.35
        elif gear == 2: # 가속 주행
            base_speed = 0.55
        elif gear ==1: # 감속 주행
            base_speed = 0.25
        
        steering = yaw_k*yaw + np.arctan2(k*error,base_speed)
        self.steer = steering

        speed = base_speed #steering에 따른 속도 조절이 필요?

        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'line_follow'
        msg.drive.steering_angle = steering
        msg.drive.speed = speed

        self.ack_pub.publish(msg)

    
    def sliding_window(self,img,n_windows=10,margin = 12,minpix = 5):
        y = img.shape[0]
        x = img.shape[1]

        histogram = np.sum(img[y//2:,:], axis=0)   
        midpoint = int(histogram.shape[0]/2)
        leftx_current = np.argmax(histogram[:midpoint])
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

            win_xll = leftx_current - margin  # 녹색사각형 크기 : 가로 24, 세로 26
            win_xlh = leftx_current + margin
            win_xrl = rightx_current - margin
            win_xrh = rightx_current + margin

            cv.rectangle(out_img,(win_xll,win_yl),(win_xlh,win_yh),(0,255,0), 2) 
            cv.rectangle(out_img,(win_xrl,win_yl),(win_xrh,win_yh),(0,255,0), 2) 

            # 슬라이딩 윈도우 박스(녹색박스) 하나 안에 있는 흰색 픽셀의 x좌표를 모두 모은다.
            good_left_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xll)&(nz[1] < win_xlh)).nonzero()[0]
            good_right_inds = ((nz[0] >= win_yl)&(nz[0] < win_yh)&(nz[1] >= win_xrl)&(nz[1] < win_xrh)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # 구한 x좌표 리스트에서 흰색점이 5개 이상인 경우에 한해 x 좌표의 평균값을 구함. -> 이 값을 슬라이딩 윈도우의 중심점으로 사용
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nz[1][good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = int(np.mean(nz[1][good_right_inds]))

            lx.append(leftx_current)
            ly.append((win_yl + win_yh)/2)

            rx.append(rightx_current)
            ry.append((win_yl + win_yh)/2)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # 슬라이딩 윈도우의 중심점(x좌표) 9개를 가지고 2차 함수를 만들어낸다.    
        lfit = np.polyfit(np.array(ly),np.array(lx),1)
        rfit = np.polyfit(np.array(ry),np.array(rx),1)

        # 왼쪽과 오른쪽 각각 파란색과 빨간색으로 색상 변경
        out_img[nz[0][left_lane_inds], nz[1][left_lane_inds]] = [255, 0, 0]
        out_img[nz[0][right_lane_inds] , nz[1][right_lane_inds]] = [0, 0, 255]
        cv.imshow("viewer", out_img)
        self.debug_publisher1.publish(self.cv_bridge.cv2_to_imgmsg(out_img))
        
        return lfit, rfit

    def cal_center_line(self, lfit, rfit):
        """
        lfit, rfit : np.polyfit으로 구한 왼쪽/오른쪽 차선의 1차 다항식 계수
                     x = a*y + b 형태 (len == 2)
        반환값:
            yaw   : 중앙 차선의 진행 방향 각도 (라디안)
            error : 차량(이미지 중앙) 기준, 차선 중앙의 x 오프셋(px)
        """

        # 1) 왼쪽/오른쪽 차선의 계수를 평균내서 '중앙선' 계수로 사용
        cfit = (lfit + rfit) / 2.0  # [a, b]

        # 2) 중앙선을 평가할 y 위치 선택
        #    - 보통 이미지 아래쪽 3/4 지점에서 방향과 오프셋을 보게 함
        if self.filtered_img is not None:
            h, w = self.filtered_img.shape[:2]
        else:
            # fallback (warp 이미지 크기 기준)
            h, w = 160, 640

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


    def draw_lane(self,image, warp_roi,warp_img0, inv_mat, left_fit, right_fit):
            """
            image    : 원본 BGR 이미지
            warp_roi : ROI만 잘라낸 warp 이미지 (self.warp_img)
            inv_mat  : self.inv_warp_mat
            left_fit, right_fit : ROI 좌표계 기준 polyfit 결과
            """

            # 1) 전체 warp 이미지 (ROI 자르기 전)
            if warp_img0 is not None:
                base_warp = warp_img0
            else:
                # 혹시라도 warp_img0가 없으면, 기존처럼 ROI 전체를 기준으로라도 그리기
                base_warp = warp_roi

            full_h, full_w = base_warp.shape[:2]
            roi_h, roi_w   = warp_roi.shape[:2]

            # ROI가 항상 아래쪽을 자르고 있으니까:
            #   예) full_h = 480, roi_h = 170 -> offset = 480 - 170 = 310
            roi_offset_y = full_h - roi_h

            # 2) ROI 좌표계에서 중앙선 그릴 y축
            yMax = roi_h
            ploty = np.linspace(0, yMax - 1, yMax)

            # ROI 기준 x좌표
            left_fitx  = left_fit[0] * ploty + left_fit[1] 
            right_fitx = right_fit[0] * ploty + right_fit[1] 

            # 3) ROI y좌표를 전체 warp 좌표계로 올려 붙이기
            ploty_full = ploty + roi_offset_y  # 여기서가 핵심!

            pts_left  = np.array([np.transpose(np.vstack([left_fitx,  ploty_full]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty_full])))])

            pts = np.hstack((pts_left, pts_right))

            # 4) 전체 warp 크기의 빈 컬러 이미지 만들고 lane area 채우기
            color_warp = np.zeros_like(base_warp).astype(np.uint8)  # full_h x full_w

            cv.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
            cv.imshow('debug',color_warp)
            # 5) 역원근변환으로 원본 이미지 좌표계로 되돌리고 오버레이
            newwarp = cv.warpPerspective(color_warp, inv_mat, (image.shape[1], image.shape[0]))
            result = cv.addWeighted(image, 1, newwarp, 0.3, 0)
            
            text1 = f"yaw: {self.yaw:.3f} rad ({self.steer:.1f} deg)"
            text2 = f"err: {self.error:.1f} px"
            #디버깅 텍스트 추가
            cv.putText(result, text1, (30, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(result, text2, (30, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv.LINE_AA)

            return result

    
    def main(self):
        if self.start_mission == 0:
            rospy.loginfo("mission start!!! / Lane Following is always working...")
            self.start_mission = 1.
            rospy.loginfo('------mission 1------')
            msg = Float64()
            msg.data = 1.0
            self.mission_pub.publish(msg)
        if self.bgr is None:
            return
        self.zone = None
        self.warp_img0 = self.warpping(self.bgr) 
        self.warp_img = self.roi_set(self.warp_img0)       
        g_filtered = self.Gaussian_filter(self.warp_img)
        
        self.roi_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(g_filtered, "bgr8"))
        self.gear = 3
        self.white_img = self.white_color_filter_hsv(g_filtered)
        
        
        self.filtered_img = self.binary_filter(self.white_img)     
        self.binary_img_pub.publish(self.cv_bridge.cv2_to_imgmsg(self.filtered_img))

        lfit, rfit = self.sliding_window(self.filtered_img)
        #
        # print(f'lfit: {lfit}, rfit: {rfit}')

        self.yaw,self.error = self.cal_center_line(lfit,rfit)

        self.cal_steering(yaw=self.yaw,error=self.error)
        
        #디버깅2: 원본 이미지에 차로 영역 반투명하게 채워넣기
        debug2_img = self.draw_lane(self.bgr,self.warp_img,self.warp_img0,self.inv_warp_mat,lfit,rfit)
        self.debug_publisher2.publish(self.cv_bridge.cv2_to_imgmsg(debug2_img))
        cv.imshow('debug',debug2_img)
        cv.imshow('2 roi_warp',self.warp_img)
        #cv.imshow('2white_filter',self.white_img)
        #cv.imshow('3 Gaussian filltered image',g_filtered)
        cv.imshow('Binary_img',self.filtered_img)

        
        cv.waitKey(1)



if __name__ == '__main__':
    try:
        lf = LaneFollow()
        rate = rospy.Rate(30)  # 30Hz

        while not rospy.is_shutdown():
            lf.main()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
