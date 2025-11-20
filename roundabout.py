#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker


class Mission5Roundabout:
    def __init__(self):
        rospy.loginfo("=== Mission 5: Roundabout (Dynamic-Car Only) ===")

        # ---------------------------------------------------------
        # ROUNDABOUT PARAMETERS (miniature)
        # ---------------------------------------------------------
        # Lidar forward direction = 180 deg
        self.FRONT_CENTER = math.pi  # 180°

        # Very narrow conflict zone (12°)
        self.ROI_HALF_WIDTH = math.radians(12)

        # Distances suitable for miniature roundabout
        self.BLOCK_DIST = 0.35     # too close = danger
        self.SAFE_DIST  = 0.55     # gap begins after car passes

        # Velocity thresholds
        # Static objects: |vel| < 0.02 m/s (ignore)
        self.STATIC_VELOCITY = 0.02
        self.APPROACH_VEL = -0.03   # approaching vehicle

        # Time logic
        self.MIN_CLEAR_TIME = 1.0
        self.COMMIT_TIME = 3.0

        # ERPM and steering
        self.ERPM_STOP = 0.0
        self.ERPM_WAIT = 0.0
        self.ERPM_GO   = 1200.0       # safe accelerate into roundabout
        self.STEER_CENTER = 0.5      # straight

        # State variables
        self.mode = "WAIT"
        self.seen_first_car = False

        self.last_dist = None
        self.last_time = None
        self.clear_start_time = None
        self.go_start_time = None

        # ---------------------------------------------------------
        # ROS I/O
        # ---------------------------------------------------------
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)

        self.speed_pub = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steer_pub = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)

        self.marker_pub = rospy.Publisher("/mission5_marker", Marker, queue_size=1)
        self.text_pub   = rospy.Publisher("/mission5_text", Marker, queue_size=1)

        rospy.Timer(rospy.Duration(0.05), self.update)  # 20 Hz


    # ======================================================
    # LIDAR processing
    # ======================================================
    def scan_callback(self, scan):
        self.scan = scan

    def get_roi(self):
        """Return nearest moving object distance + velocity OR (None,None)."""

        if not hasattr(self, "scan"):
            return None, None

        # Extract ranges and angles
        ranges = np.array(self.scan.ranges, dtype=float)
        angles = self.scan.angle_min + np.arange(len(ranges)) * self.scan.angle_increment

        mask_valid = np.isfinite(ranges)
        ranges = ranges[mask_valid]
        angles = angles[mask_valid]

        # Angular difference from front (180°)
        diff = np.arctan2(
            np.sin(angles - self.FRONT_CENTER),
            np.cos(angles - self.FRONT_CENTER)
        )

        # Filter ROI
        roi = ranges[np.abs(diff) < self.ROI_HALF_WIDTH]

        if len(roi) == 0:
            self.last_dist = None
            self.last_time = rospy.Time.now().to_sec()
            return None, None

        nearest = float(np.min(roi))
        now = rospy.Time.now().to_sec()

        # Compute velocity
        vel = None
        if self.last_dist is not None and self.last_time is not None:
            dt = now - self.last_time
            if dt > 1e-3:
                vel = (nearest - self.last_dist) / dt

        self.last_dist = nearest
        self.last_time = now

        # -----------------------------------------------
        # IGNORE STATIC OBJECTS
        # -----------------------------------------------
        # If it doesn’t move → it is a wall, cone, barrier, etc.
        if vel is not None and abs(vel) < self.STATIC_VELOCITY:
            return None, None

        # Only dynamic objects matter
        return nearest, vel


    # ======================================================
    # RVIZ VISUALIZATION
    # ======================================================
    def show_marker(self, dist, color, text):
        # Sphere
        m = Marker()
        m.header.frame_id = "laser"
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color.r, m.color.g, m.color.b = color
        m.color.a = 1.0

        x = 0.5 if dist is None else dist
        m.pose.position.x = -x  # flip for front-facing
        m.pose.position.y = 0
        m.pose.position.z = 0
        self.marker_pub.publish(m)

        # Text
        t = Marker()
        t.header.frame_id = "laser"
        t.type = Marker.TEXT_VIEW_FACING
        t.scale.z = 0.25
        t.color.r = t.color.g = t.color.b = 1.0
        t.color.a = 1.0
        t.pose.position.x = 0
        t.pose.position.y = 0
        t.pose.position.z = 0.5
        t.text = text
        self.text_pub.publish(t)


    # ======================================================
    # MAIN LOOP
    # ======================================================
    def update(self, event):

        # Keep steering straight
        self.steer_pub.publish(Float64(self.STEER_CENTER))

        dist, vel = self.get_roi()

        if self.mode == "WAIT":
            self.wait_mode(dist, vel)
        elif self.mode == "GO":
            self.go_mode()


    # ======================================================
    # WAIT MODE
    # ======================================================
    def wait_mode(self, dist, vel):

        self.speed_pub.publish(Float64(self.ERPM_WAIT))

        # No dynamic object detected
        if dist is None:
            # Must see first car before entering
            if not self.seen_first_car:
                self.show_marker(None, (1,1,0), "WAIT (must see car)")
                return

            # Now performing gap detection
            if self.clear_start_time is None:
                self.clear_start_time = rospy.Time.now().to_sec()

            t_clear = rospy.Time.now().to_sec() - self.clear_start_time
            self.show_marker(None, (0,1,0), f"CLEAR {t_clear:.1f}s")

            if t_clear > self.MIN_CLEAR_TIME:
                self.start_go()
            return

        # A moving object is detected here
        self.seen_first_car = True
        self.clear_start_time = None

        # Too close → WAIT
        if dist < self.BLOCK_DIST:
            self.show_marker(dist, (1,0,0), f"WAIT close d={dist:.2f}")
            return

        # Approaching → WAIT
        if vel is not None and vel < self.APPROACH_VEL:
            self.show_marker(dist, (1,0,0), f"APPROACH d={dist:.2f}")
            return

        # Dynamic car present but moving away or stable → form gap
        if self.clear_start_time is None:
            self.clear_start_time = rospy.Time.now().to_sec()

        t_clear = rospy.Time.now().to_sec() - self.clear_start_time
        self.show_marker(dist, (0,1,0), f"GAP {t_clear:.1f}s")

        if t_clear > self.MIN_CLEAR_TIME:
            self.start_go()


    # ======================================================
    # GO MODE
    # ======================================================
    def start_go(self):
        rospy.loginfo("[Mission5] GO — Gap confirmed.")
        self.mode = "GO"
        self.go_start_time = rospy.Time.now().to_sec()

    def go_mode(self):
        t = rospy.Time.now().to_sec() - self.go_start_time
        self.show_marker(0.5, (0,0,1), f"GO {t:.1f}s")
        self.speed_pub.publish(Float64(self.ERPM_GO))

        if t > self.COMMIT_TIME:
            rospy.loginfo("[Mission5] Commit done → STOP.")
            self.speed_pub.publish(Float64(self.ERPM_STOP))


if __name__ == "__main__":
    rospy.init_node("roundabout")
    Mission5Roundabout()
    rospy.spin()

