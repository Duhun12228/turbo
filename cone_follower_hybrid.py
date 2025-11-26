#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float64
import time


class ConeCenterlineFollower:
    def __init__(self):
        rospy.init_node("cone_centerline_follower")

        # ==============================
        # Parameters (기존)
        # ==============================

        self.ADD_PI = True

        self.ROI_X_MIN = 0.20
        self.ROI_X_MAX = 0.6  # 원래 값 유지
        self.ROI_Y_MIN = -0.5
        self.ROI_Y_MAX =  0.5

        self.X_SAMPLES = [0.20, 0.30, 0.45, 0.60]
        self.DX_WINDOW = 0.20

        self.DEFAULT_HALF_LANE = 0.35
        self.MIN_HALF_LANE     = 0.2
        self.MAX_HALF_LANE     = 0.50

        self.WHEELBASE = 0.28
        self.MAX_STEER = math.radians(40.0)
        self.LOOKAHEAD = 0.30

        self.BASE_SPEED    = 0.25
        self.MIN_SPEED     = 0.2
        self.SLOWDOWN_GAIN = 0.03

        self.ERPM_BASE = 1500.0

        self.STEER_SMOOTH_ALPHA = 0.08
        self.SPEED_SMOOTH_ALPHA = 0.3
        self.last_steer = 0.0
        self.last_speed = 0.0

        self.last_center_y = 0.0
        self.last_lane_half = self.DEFAULT_HALF_LANE
        self.last_valid_pair = False
        self.start_time = 0
        self.mission_fallback_time = 0

        # ==============================
        # 전방 장애물 → 후진 관련 파라미터 (NEW)
        # ==============================

        self.FRONT_ANGLE_MARGIN = math.radians(15.0)  # ±60도 = 총 120도
        self.FRONT_STOP_DIST = 0.35              # 25cm
        self.REVERSE_SPEED = -0.2                    # m/s (후진이니까 음수로 publish)
        self.REVERSE_TIME = 0.5               # 1초 동안 후진
        self.reverse_start_time = 0.0
        self.front_blocked = False                    # 전방 25cm 이내 장애물 여부 flag

        # ==============================
        # ROS I/O
        # ==============================

        rospy.Subscriber("/scan", LaserScan, self.lidar_cb, queue_size=1)

        # RViz publishers
        self.raw_pub    = rospy.Publisher("/cone_center_raw",   Marker, queue_size=1)
        self.left_pub   = rospy.Publisher("/cone_left_points",  Marker, queue_size=1)
        self.right_pub  = rospy.Publisher("/cone_right_points", Marker, queue_size=1)
        self.path_pub   = rospy.Publisher("/cone_center_path",  Marker, queue_size=1)
        self.target_pub = rospy.Publisher("/cone_center_target",Marker, queue_size=1)

        self.ack_pub_1 = rospy.Publisher(
            '/high_level/ackermann_cmd_mux/input/nav_1',
            AckermannDriveStamped,
            queue_size=10
        )

        self.mission_sub = rospy.Subscriber(
            '/mission_num', Float64, self.mission_cb, queue_size=1
        )
        self.mission_pub = rospy.Publisher('/mission_num', Float64, queue_size=10)

        self.state = None
        self.xs_roi = None
        self.ys_roi = None
        self.left_pts = []
        self.right_pts = []
        self.path = None
        
        self.current_mission = 3.0

    # ==============================
    # Mission
    # ==============================

    def mission_cb(self,msg):
        self.current_mission = msg.data

    # ==============================
    # LiDAR callback
    # ==============================

    def lidar_cb(self, scan: LaserScan):
        ranges = np.array(scan.ranges, dtype=float)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]

        if self.ADD_PI:
            angles = angles + math.pi

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        # ---- 전방(±60도) 25cm 내 장애물 체크 (NEW) ----
        # 차량 좌표계에서 각도 계산 (앞이 0 rad라고 가정)
        thetas = np.arctan2(ys, xs)
        dists  = np.hypot(xs, ys)

        front_mask = np.abs(thetas) < self.FRONT_ANGLE_MARGIN
        if np.any(front_mask):
            front_dists = dists[front_mask]
            self.front_blocked = np.any(front_dists < self.FRONT_STOP_DIST)
        else:
            self.front_blocked = False

        # ---- 기존 ROI 처리 ----
        mask = (
            (xs > self.ROI_X_MIN) & (xs < self.ROI_X_MAX) &
            (ys > self.ROI_Y_MIN) & (ys < self.ROI_Y_MAX)
        )

        self.xs_roi = xs[mask]
        self.ys_roi = ys[mask]

        # Emergency turn (기존 그대로 둠)
        d = np.hypot(self.xs_roi, self.ys_roi)
        if len(d) > 0:
            min_d = np.min(d)
            if min_d < 0.25:
                idx = np.argmin(d)
                cone_y = self.ys_roi[idx]
                if cone_y > 0:
                    emergency = -self.MAX_STEER
                else:
                    emergency = self.MAX_STEER
                self.publish_drive(emergency, self.MIN_SPEED)
                rospy.logwarn("EMERGENCY AVOIDANCE TURN!")
                self.publish_markers(self.xs_roi, self.ys_roi, [], None)
                return

        if len(self.xs_roi) == 0:
            self.left_pts  = np.empty((0, 2))
            self.right_pts = np.empty((0, 2))
            self.path = None
            self.publish_markers(self.xs_roi, self.ys_roi, [], None)
            return            

        left_mask  = self.ys_roi > 0.15
        right_mask = self.ys_roi < -0.15

        self.left_pts  = np.vstack((self.xs_roi[left_mask],  self.ys_roi[left_mask])).T if np.any(left_mask) else np.empty((0, 2))
        self.right_pts = np.vstack((self.xs_roi[right_mask], self.ys_roi[right_mask])).T if np.any(right_mask) else np.empty((0, 2))

        self.path = self.compute_centerline_path(self.left_pts, self.right_pts)

    # ==============================
    # Centerline logic (기존)
    # ==============================

    def compute_centerline_path(self, left_pts, right_pts):
        path = []
        for x_i in self.X_SAMPLES:
            yl = None
            yr = None

            if len(left_pts) > 0:
                m = np.abs(left_pts[:, 0] - x_i) < self.DX_WINDOW
                cand = left_pts[m]
                if len(cand) > 0:
                    yl = cand[np.argmin(np.abs(cand[:, 0] - x_i)), 1]

            if len(right_pts) > 0:
                m = np.abs(right_pts[:, 0] - x_i) < self.DX_WINDOW
                cand = right_pts[m]
                if len(cand) > 0:
                    yr = cand[np.argmin(np.abs(cand[:, 0] - x_i)), 1]

            if yl is not None and yr is not None:
                lane_width = yl - yr
                half_lane = lane_width * 0.5
                if not (self.MIN_HALF_LANE <= half_lane <= self.MAX_HALF_LANE):
                    half_lane = self.DEFAULT_HALF_LANE
                center_y = 0.5 * (yl + yr)
                self.last_center_y = center_y
                self.last_lane_half = half_lane
                self.last_valid_pair = True

            elif self.last_valid_pair:
                center_y = self.last_center_y
            else:
                center_y = 0.0

            path.append((x_i, center_y))

        return path

    # ==============================
    # Path smoothing (기존)
    # ==============================

    def smooth_path(self, pts):
        if len(pts) < 3:
            return pts
        pts = np.array(pts)
        y = pts[:,1]
        for _ in range(1):
            y[1:-1] = 0.25*y[:-2] + 0.5*y[1:-1] + 0.25*y[2:]
        pts[:,1] = y
        return [tuple(p) for p in pts]

    # ==============================
    # Pure pursuit (기존)
    # ==============================

    def pure_pursuit(self, path):
        pts = np.array(path)
        d = np.linalg.norm(pts, axis=1)
        idx = np.argmin(np.abs(d - self.LOOKAHEAD))
        if idx < 0 or idx >= len(pts):
            return 0.0, self.MIN_SPEED, None
        
        tx, ty = pts[idx]
        alpha = math.atan2(ty, tx)
        delta = math.atan2(2*self.WHEELBASE*math.sin(alpha), self.LOOKAHEAD)
        delta = max(-self.MAX_STEER, min(self.MAX_STEER, delta))
        turn_ratio = abs(delta) / self.MAX_STEER
        speed = self.BASE_SPEED - self.SLOWDOWN_GAIN * turn_ratio
        speed = max(self.MIN_SPEED, speed)
        return delta, speed, (tx, ty)

    # ==============================
    # EXP smooth (기존)
    # ==============================

    def exp_smooth(self, prev, new, alpha):
        return alpha*prev + (1-alpha)*new

    # ==============================
    # Ackermann driving (기존)
    # ==============================

    def publish_drive(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.steering_angle = float(steer)   # radians
        msg.drive.speed = float(speed)            # m/s
        self.ack_pub_1.publish(msg)

    # ==============================
    # RViz markers (기존)
    # ==============================

    def publish_markers(self, xs_roi, ys_roi, path, target,
                        left_pts=None, right_pts=None):

        raw = Marker()
        raw.header.frame_id = "laser"
        raw.type = Marker.POINTS
        raw.scale.x = raw.scale.y = 0.03
        raw.color.r = 1.0
        raw.color.a = 1.0
        raw.pose.orientation.w = 1.0
        for x, y in zip(xs_roi, ys_roi):
            raw.points.append(Point(x=-x, y=-y, z=0))
        self.raw_pub.publish(raw)

        if left_pts is not None:
            m = Marker()
            m.header.frame_id = "laser"
            m.type = Marker.POINTS
            m.scale.x = m.scale.y = 0.05
            m.color.b = 1.0
            m.color.a = 1.0
            m.pose.orientation.w = 1.0
            for x,y in left_pts:
                m.points.append(Point(x=-x, y=-y, z=0))
            self.left_pub.publish(m)

        if right_pts is not None:
            m = Marker()
            m.header.frame_id = "laser"
            m.type = Marker.POINTS
            m.scale.x = m.scale.y = 0.05
            m.color.g = 1.0
            m.color.a = 1.0
            m.pose.orientation.w = 1.0
            for x,y in right_pts:
                m.points.append(Point(x=-x, y=-y, z=0))
            self.right_pub.publish(m)

        pm = Marker()
        pm.header.frame_id = "laser"
        pm.type = Marker.POINTS
        pm.scale.x = pm.scale.y = 0.06
        pm.color.r = 0.0
        pm.color.g = 1.0
        pm.color.b = 1.0
        pm.color.a = 1.0
        pm.pose.orientation.w = 1.0
        for x,y in path:
            pm.points.append(Point(x=-x, y=-y, z=0))
        self.path_pub.publish(pm)

        t = Marker()
        t.header.frame_id = "laser"
        t.type = Marker.SPHERE
        t.scale.x = t.scale.y = t.scale.z = 0.16
        t.color.r = 1.0
        t.color.a = 1.0
        t.pose.orientation.w = 1.0
        if target:
            tx, ty = target
            t.pose.position.x = -tx
            t.pose.position.y = -ty
        else:
            t.action = Marker.DELETE
        self.target_pub.publish(t)

    # ==============================
    # Main state machine
    # ==============================

    def main(self):
            # ---- 0) 이미 후진 중인 상태 처리 (front_blocked와 무관) ----
            if self.state == 'reverse':
                elapsed = time.time() - self.reverse_start_time
                if elapsed < self.REVERSE_TIME:
                    # 후진도 부드럽게
                    speed_cmd = self.exp_smooth(self.last_speed, self.REVERSE_SPEED, self.SPEED_SMOOTH_ALPHA)
                    self.last_speed = speed_cmd
                    self.publish_drive(-self.last_steer, speed_cmd)
                else:
                    # 후진 끝 → 정지 명령만 보내고, last_speed는 건드리지 말기
                    self.publish_drive(0.0, 0.0)
                    # self.last_speed = 0.0   # <<< 이 줄 지워
                    self.state = None
                    self.front_blocked = False
                return

            if self.front_blocked and self.state != 'reverse':
                rospy.logwarn("FRONT BLOCKED -> REVERSE PHASE")
                self.state = 'reverse'
                self.reverse_start_time = time.time()
                self.reverse_start_time = time.time()
                # 여기에서 바로 self.publish_drive(...) 하지 말고,
                # 다음 루프에서 위의 state=='reverse' 블록이 처리하게 둠
                return



            # ---- 2) LiDAR가 아직 준비 안 됐으면 ----
            if self.path is None:
                rospy.loginfo('cone111 is not detected')
                return 
            
            if self.xs_roi is None or self.ys_roi is None:
                rospy.loginfo('cone222 is not detected')
                return

            # ------------------------------
            # STATE: 초기 (None)
            # ------------------------------
            if self.state is None:

                if len(self.path) == 0:
                    self.publish_markers(self.xs_roi, self.ys_roi, [], None)
                    rospy.loginfo('cone333 is not detected')
                    return
                
                # 콘 보이는 경우
                self.path = self.smooth_path(self.path)

                steer_raw, speed_raw, target = self.pure_pursuit(self.path)

                if target is None:
                    self.publish_markers(self.xs_roi, self.ys_roi, self.path, None)
                    return

                steer = self.exp_smooth(self.last_steer, steer_raw, self.STEER_SMOOTH_ALPHA)
                speed = self.exp_smooth(self.last_speed, speed_raw, self.SPEED_SMOOTH_ALPHA)
                self.last_steer = steer
                self.last_speed = speed

                self.publish_drive(steer, speed)
                self.publish_markers(self.xs_roi, self.ys_roi, self.path, target, self.left_pts, self.right_pts)
                self.state = 'cone_following'
                rospy.loginfo('cone is detected@@@')
                
                self.mission_fallback_time = time.time()
            
            # ------------------------------
            # STATE: 콘 따라가는 중
            # ------------------------------
            elif self.state == 'cone_following':

                if len(self.xs_roi) == 0:
                    self.publish_markers(self.xs_roi, self.ys_roi, [], None)
                    if time.time() - self.mission_fallback_time < 3.0:
                        rospy.loginfo('cone is not detected but maybe some error') 
                        self.state = None
                        return
                    else:
                        self.state = 'Done'
                        return
                    
                rospy.loginfo('Following Cone@!!!')

                self.path = self.smooth_path(self.path)

                steer_raw, speed_raw, target = self.pure_pursuit(self.path)

                if target is None:
                    self.publish_markers(self.xs_roi, self.ys_roi, self.path, None)
                    return

                steer = self.exp_smooth(self.last_steer, steer_raw, self.STEER_SMOOTH_ALPHA)
                speed = self.exp_smooth(self.last_speed, speed_raw, self.SPEED_SMOOTH_ALPHA)
                self.last_steer = steer
                self.last_speed = speed

                self.publish_drive(steer, speed)
                self.publish_markers(self.xs_roi, self.ys_roi, self.path, target, self.left_pts, self.right_pts)
            
            # ------------------------------
            # STATE: Done
            # ------------------------------
            elif self.state == 'Done':
                rospy.loginfo('Cone follow is done...')

                msg = Float64()
                msg.data = 4.
                self.current_mission = 4.
                self.mission_pub.publish(msg)
                rospy.loginfo('------mission 4------')


if __name__ =='__main__':
    try:
        cf = ConeCenterlineFollower()
        rate = rospy.Rate(35)  # 25Hz

        while not rospy.is_shutdown():
            if cf.current_mission == 3.0:
                if cf.start_time == 0:
                    cf.start_time = time.time()

                elif time.time() - cf.start_time > 5.0: # 5초 동안은 그냥 기다림
                    #rospy.loginfo('cone.py is running')
                    cf.main()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
