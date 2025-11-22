#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from ackermann_msgs.msg import AckermannDriveStamped
import math
import time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class TunnelFollower50cm:
    def __init__(self):
        rospy.init_node('tunnel_follower_50cm')
        # Tunnel geometry
        self.TUNNEL_WIDTH = 0.60
        self.TUNNEL_HALF = self.TUNNEL_WIDTH / 2.0

        # Region Of Interest
        self.ROI_X_MIN = 0.20
        self.ROI_X_MAX = 1.00
        self.ROI_Y_MAX = 0.50

        # Vehicle parameters
        self.WHEELBASE = 0.28
        self.MAX_STEER = math.radians(45.0)
        self.LOOKAHEAD = 0.20

        # Speed → ERPM
        self.BASE_SPEED = 0.35
        self.MIN_SPEED = 0.25

        self.scan_data = None

        # ROS I/O
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.ack_pub_1 = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=10)
        self.mission_sub = rospy.Subscriber(
            '/mission_num', Float64, self.mission_cb, queue_size=1)
        self.mission_pub = rospy.Publisher('/mission_num', Float64, queue_size=10)


        self.viz_pub = rospy.Publisher("/tunnel_wall_markers", MarkerArray, queue_size=1)
        self.target_pub = rospy.Publisher("/tunnel_target_point", Marker, queue_size=1)

        self.current_mission = 0.0
        self.state = None
        

    def scan_callback(self, msg):
        self.scan_data = msg

    def mission_cb(self, msg):
        self.current_mission = msg.data


    # ================================================================
    # Extract wall points (with correct LiDAR orientation)
    # ================================================================
    def extract_wall_points(self, scan):

        ranges = np.array(scan.ranges, dtype=float)
        angles_orig = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles_orig = angles_orig[valid]

        # 🔥 FIX: SAME ROTATION YOUR TEAMMATE USES (+180 degrees)
        angles = angles_orig + math.pi

        # Convert to Cartesian
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # ROI
        mask = (
            (x > self.ROI_X_MIN) &
            (x < self.ROI_X_MAX) &
            (np.abs(y) < self.ROI_Y_MAX)
        )

        x_roi = x[mask]
        y_roi = y[mask]

        left_mask = y_roi > 0.0
        right_mask = y_roi < 0.0

        left_pts = np.column_stack((x_roi[left_mask], y_roi[left_mask]))
        right_pts = np.column_stack((x_roi[right_mask], y_roi[right_mask]))

        return left_pts, right_pts


    def fit_line(self, pts):
        if pts is None or len(pts) < 5:
            return None
        try:
            x = pts[:, 0]
            y = pts[:, 1]
            a, b = np.polyfit(x, y, 1)
            return float(a), float(b)
        except:
            return None


    def compute_center_target(self, left_line, right_line, left_pts, right_pts):

        x = self.LOOKAHEAD

        if left_line and right_line:
            aL, bL = left_line
            aR, bR = right_line
            yL = aL * x + bL
            yR = aR * x + bR
            return np.array([x, 0.5 * (yL + yR)])

        if left_line:
            aL, bL = left_line
            yL = aL * x + bL
            return np.array([x, yL - self.TUNNEL_HALF])

        if right_line:
            aR, bR = right_line
            yR = aR * x + bR
            return np.array([x, yR + self.TUNNEL_HALF])

        return None


    def pure_pursuit(self, target):

        tx, ty = float(target[0]), float(target[1])

        Ld = math.sqrt(tx*tx + ty*ty)
        if Ld < 1e-4:
            return 0.0, self.MIN_SPEED

        alpha = math.atan2(ty, tx)

        delta = math.atan2(2.0 * self.WHEELBASE * math.sin(alpha), Ld)
        delta = max(-self.MAX_STEER, min(self.MAX_STEER, delta))

        turn_ratio = abs(delta) / self.MAX_STEER
        speed = self.BASE_SPEED - 0.10 * turn_ratio
        speed = max(self.MIN_SPEED, speed)

        return delta, speed


    # ================================================================
    # 🚀 CORRECT SERVO MAPPING (THIS FIXES EVERYTHING)
    # ================================================================



    # ================================================================
    # Visualization
    # ================================================================
    def publish_visualization(self, left_pts, right_pts, target):

        ma = MarkerArray()
        now = rospy.Time.now()
        frame = "laser"
        mid = 0

        def add_marker(m):
            nonlocal mid
            m.id = mid
            m.header.stamp = now
            m.lifetime = rospy.Duration(0.2)
            ma.markers.append(m)
            mid += 1

        # Left points (green)
        if left_pts is not None and len(left_pts) > 0:
            m = Marker()
            m.header.frame_id = frame
            m.type = Marker.POINTS
            m.scale.x = m.scale.y = 0.03
            m.color.g = 1.0
            m.color.a = 1.0
            for x, y in left_pts:
                m.points.append(Point(x=x, y=y, z=0.0))
            add_marker(m)

        # Right points (blue)
        if right_pts is not None and len(right_pts) > 0:
            m = Marker()
            m.header.frame_id = frame
            m.type = Marker.POINTS
            m.scale.x = m.scale.y = 0.03
            m.color.b = 1.0
            m.color.a = 1.0
            for x, y in right_pts:
                m.points.append(Point(x=x, y=y, z=0.0))
            add_marker(m)

        self.viz_pub.publish(ma)

        if target is not None:
            m = Marker()
            m.header.frame_id = frame
            m.type = Marker.SPHERE
            m.scale.x = m.scale.y = m.scale.z = 0.10
            m.color.r = 1.0
            m.color.b = 1.0
            m.color.a = 1.0
            m.pose.position.x = target[0]
            m.pose.position.y = target[1]
            m.pose.position.z = 0.0
            self.target_pub.publish(m)

    def publish_ack(self, speed, steering = 0.0):
        ack = AckermannDriveStamped()
        ack.header.stamp = rospy.Time.now()
        ack.drive.speed = speed
        ack.drive.steering_angle = steering
        self.ack_pub_1.publish(ack)

    def main(self):
        if self.scan_data is None:
            return

        left_pts, right_pts = self.extract_wall_points(self.scan_data)

        left_line = self.fit_line(left_pts)
        right_line = self.fit_line(right_pts)
        
        if self.state == None:
            if left_line is None and right_line is None:
                self.publish_visualization(left_pts, right_pts, None)
                rospy.loginfo("No line detected")
                return
            else:
                target = self.compute_center_target(left_line, right_line, left_pts, right_pts)
                self.state = 'tunnel_detected'
                steering, speed = self.pure_pursuit(target)
                self.publish_ack(speed, steering)
                self.publish_visualization(left_pts, right_pts, target)

                rospy.loginfo('tunnel_detected!! follow start')
                self.start_time = time.time()

        elif self.state == 'tunnel_detected':
            if left_line is None and right_line is None:
                if time.time() - self.start_time < 3.0:
                    rospy.loginfo('tunnel is not detected but might be some error')
                    return
                else:
                    self.state = 'Done'
                    return
            else:
                target = self.compute_center_target(left_line, right_line, left_pts, right_pts)
                steering, speed = self.pure_pursuit(target)
                self.publish_visualization(left_pts, right_pts, target)
                self.publish_ack(speed, steering)

        elif self.state == 'Done':
            msg = Float64()
            msg.data = 5.
            self.current_mission = 5.
            self.mission_pub.publish(msg)
            rospy.loginfo('------mission 5------')


        
if __name__ =='__main__':
    try:
        tf = TunnelFollower50cm()
        rate = rospy.Rate(25)  # 25Hz

        while not rospy.is_shutdown():
            if tf.current_mission == 4.0:
                tf.main()
                rate.sleep()

    except rospy.ROSInterruptException:
        pass
