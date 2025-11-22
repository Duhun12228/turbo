#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image


class Mission5Roundabout:
    def __init__(self):
        rospy.loginfo("=== Mission 5: Roundabout (Dynamic + Must See First Car + Multi-Car Safe) ===")
        self.cv_bridge = CvBridge()

        # ---------------------------------------------------------
        # Miniature roundabout parameters
        # ---------------------------------------------------------
        self.FRONT_CENTER = math.pi                # 180° = forward
        self.ROI_HALF_WIDTH = math.radians(15)     # small window
        self.BLOCK_DIST = 0.35                     # very close car
        self.SAFE_DIST = 0.55                      # car passed if > 55cm
        self.STATIC_VEL_THRESH = 0.005             # slow car still counts
        self.APPROACH_VEL = -0.03                  # approaching threshold

        # Timing parameters
        self.MIN_CLEAR_TIME = 1                  # time of clear zone before GO
        self.COMMIT_TIME = 3                     # commit to entering

        # Speed and steering
        self.ERPM_WAIT = 0.0
        self.ERPM_GO   = 1200.0
        self.ERPM_STOP = 0.0
        self.STEER_CENTER = 0.5

        # Internal state
        self.mode = "WAIT"
        self.seen_first_car = False
        self.clear_start_time = None
        self.go_start_time = None
        self.last_dist = None
        self.last_time = None

        # ROS I/O
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.speed_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steer_pub = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)
        self.marker_pub = rospy.Publisher("/mission5_marker", Marker, queue_size=1)
        self.text_pub   = rospy.Publisher("/mission5_text", Marker, queue_size=1)
        self.binary_img_sub = rospy.Subscriber('/binary_img',Image,self.binary_img_cb,queue_size=10)
        self.ack_pub_1 = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=10)

        rospy.Timer(rospy.Duration(0.05), self.update)
        self.binary_img = None

    def binary_img_cb(self,msg):
        self.binary_img = self.cv_bridge.imgmsg_to_cv2(msg)
    
    # =====================================================================
    def scan_callback(self, scan):
        self.scan = scan

    # =====================================================================
    def get_dynamic_object(self):
        """Return (dist, vel) of a dynamic object OR (None, None)."""

        if not hasattr(self, "scan"):
            return None, None

        ranges = np.array(self.scan.ranges, dtype=float)
        angles = self.scan.angle_min + np.arange(len(ranges)) * self.scan.angle_increment
        mask = np.isfinite(ranges)
        ranges = ranges[mask]
        angles = angles[mask]

        diff = np.arctan2(np.sin(angles - self.FRONT_CENTER),
                          np.cos(angles - self.FRONT_CENTER))

        # Filter ROI
        roi_ranges = ranges[np.abs(diff) < self.ROI_HALF_WIDTH]
        if len(roi_ranges) == 0:
            self.last_dist = None
            self.last_time = rospy.Time.now().to_sec()
            return None, None

        nearest = float(np.min(roi_ranges))
        now = rospy.Time.now().to_sec()

        vel = None
        if self.last_dist is not None:
            dt = now - self.last_time
            if dt > 1e-3:
                vel = (nearest - self.last_dist) / dt

        self.last_dist = nearest
        self.last_time = now

        # -------- IGNORE STATIC OBJECTS --------
        if vel is not None and abs(vel) < self.STATIC_VEL_THRESH:
            return None, None

        # Car too far → ignore (track too small)
        if nearest > 0.9:
            return None, None

        return nearest, vel

    # =====================================================================
    def show_marker(self, dist, color, text):
        """Visual debug sphere + text."""
        m = Marker()
        m.header.frame_id = "laser"
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color.r, m.color.g, m.color.b = color
        m.color.a = 1.0

        x = 0.5 if dist is None else dist
        m.pose.position.x = -x      # flip for visualization
        m.pose.position.y = 0.0
        self.marker_pub.publish(m)

        t = Marker()
        t.header.frame_id = "laser"
        t.type = Marker.TEXT_VIEW_FACING
        t.scale.z = 0.25
        t.color.r = t.color.g = t.color.b = 1.0
        t.color.a = 1.0
        t.pose.position.x = 0.0
        t.pose.position.y = 0.0
        t.pose.position.z = 0.5
        t.text = text
        self.text_pub.publish(t)

    # =====================================================================
    def update(self, event):
        dist, vel = self.get_dynamic_object()

        if self.mode == "WAIT":
            self.wait_mode(dist, vel)
        else:
            self.go_mode()

    # =====================================================================
    def wait_mode(self, dist, vel):
        self.cal_steering(gear=1)

        # ------------------------------------------------------------
        # 1) WAIT UNTIL FIRST MOVING CAR IS SEEN
        # ------------------------------------------------------------
        if not self.seen_first_car:
            if dist is not None:    # first dynamic object seen
                self.seen_first_car = True
                self.show_marker(dist, (1, 0, 0), f"FIRST CAR d={dist:.2f}")
            else:
                self.show_marker(None, (1, 1, 0), "WAIT (must see a car)")
            return

        # ------------------------------------------------------------
        # 2) AFTER FIRST CAR IS SEEN → NOW HANDLE MULTIPLE CARS
        # ------------------------------------------------------------

        # If a car is detected now → must wait
        if dist is not None:
            if vel is not None and vel < self.APPROACH_VEL:
                self.show_marker(dist, (1,0,0), f"APPROACH d={dist:.2f}")
            elif dist < self.BLOCK_DIST:
                self.show_marker(dist, (1,0,0), f"TOO CLOSE d={dist:.2f}")
            else:
                self.show_marker(dist, (1,0,0), f"CAR PRESENT d={dist:.2f}")

            self.clear_start_time = None
            return

        # ------------------------------------------------------------
        # 3) NO CARS IN ROI → possibly a GAP
        # ------------------------------------------------------------
        if self.clear_start_time is None:
            self.clear_start_time = rospy.Time.now().to_sec()

        t_clear = rospy.Time.now().to_sec() - self.clear_start_time
        self.show_marker(None, (0,1,0), f"GAP {t_clear:.2f}s")

        if t_clear >= self.MIN_CLEAR_TIME:
            self.start_go()

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
    
    def cal_steering(self,yaw=None,error=None,gear=3,k=0.005,yaw_k=1.0): #각도들은 라디안, 거리는 px값, 속도는 0~1사이 스케일값 m/s
        if gear == 3: #기본 주행
            base_speed = 0.30
        elif gear == 2: # 가속 주행
            base_speed = 0.55
        elif gear ==1: # 감속 주행
            base_speed = 0
        
        if base_speed ==0:
            msg = AckermannDriveStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'round_about'
            msg.drive.speed = base_speed
            self.ack_pub_1.publish(msg)
            return
        
        steering = yaw_k*yaw + np.arctan2(k*error,base_speed)
        self.steer = steering

        speed = base_speed #steering에 따른 속도 조절이 필요?

        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'lane_decision'
        msg.drive.steering_angle = steering
        msg.drive.speed = speed

        self.ack_pub_1.publish(msg)
    # =====================================================================
    def start_go(self):
        rospy.loginfo("[Mission5] GO — clear gap after first car.")
        self.mode = "GO"
        self.go_start_time = rospy.Time.now().to_sec()

    # =====================================================================

    def follow_right(self,img):
        rfit = self.sliding_window_right(img)
        yaw,error = self.cal_center_line(rfit)
        self.cal_steering(yaw,error)

    def go_mode(self):
        self.follow_right(self.binary_img)
        rate = rospy.Rate(25)
        rate.sleep()

        


if __name__ == "__main__":
    rospy.init_node("roundabout")
    Mission5Roundabout()
    rospy.spin()
