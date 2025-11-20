#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


class TunnelFollower50cm:
    def __init__(self):

        rospy.loginfo("=== TUNNEL FOLLOWER (50cm) WITH CORRECT STEERING MAP ===")

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
        self.BASE_SPEED = 0.1
        self.MIN_SPEED = 0.05
        self.ERPM_BASE = 2000.0

        self.scan_data = None

        # ROS I/O
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        self.speed_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steer_pub = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)

        self.viz_pub = rospy.Publisher("/tunnel_wall_markers", MarkerArray, queue_size=1)
        self.target_pub = rospy.Publisher("/tunnel_target_point", Marker, queue_size=1)

        rospy.Timer(rospy.Duration(0.05), self.control_loop)


    def scan_callback(self, msg):
        self.scan_data = msg


    def control_loop(self, event):
        if self.scan_data is None:
            return

        left_pts, right_pts = self.extract_wall_points(self.scan_data)

        left_line = self.fit_line(left_pts)
        right_line = self.fit_line(right_pts)

        if left_line is None and right_line is None:
            self.publish_drive(0.0, self.MIN_SPEED)
            self.publish_visualization(left_pts, right_pts, None)
            return

        target = self.compute_center_target(left_line, right_line, left_pts, right_pts)

        if target is None:
            self.publish_drive(0.0, self.MIN_SPEED)
            self.publish_visualization(left_pts, right_pts, None)
            return

        steering, speed = self.pure_pursuit(target)

        self.publish_drive(steering, speed)
        self.publish_visualization(left_pts, right_pts, target)


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
    def publish_drive(self, steer, speed_mps_like):

        # Convert pure pursuit output to servo (0.0–1.0)
        servo_cmd = 0.5 - (steer / self.MAX_STEER) * 0.5
        servo_cmd = max(0.0, min(1.0, servo_cmd))

        # Map speed
        ratio = speed_mps_like / self.BASE_SPEED
        erpm = self.ERPM_BASE * ratio

        # Publish to VESC
        self.steer_pub.publish(Float64(servo_cmd))
        self.speed_pub.publish(Float64(erpm))


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


if __name__ == "__main__":
    rospy.init_node("tunnel_follow_50cm", anonymous=True)
    TunnelFollower50cm()
    rospy.spin()

