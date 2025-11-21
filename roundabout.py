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
        rospy.loginfo("=== Mission 5: Roundabout (Dynamic + Must See First Car + Multi-Car Safe) ===")

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
        self.MIN_CLEAR_TIME = 1.0                  # time of clear zone before GO
        self.COMMIT_TIME = 3.0                     # commit to entering

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

        rospy.Timer(rospy.Duration(0.05), self.update)


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
        self.steer_pub.publish(Float64(self.STEER_CENTER))  # keep straight
        dist, vel = self.get_dynamic_object()

        if self.mode == "WAIT":
            self.wait_mode(dist, vel)
        else:
            self.go_mode()

    # =====================================================================
    def wait_mode(self, dist, vel):

        # Always stop in wait mode
        self.speed_pub.publish(Float64(self.ERPM_WAIT))

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

    # =====================================================================
    def start_go(self):
        rospy.loginfo("[Mission5] GO — clear gap after first car.")
        self.mode = "GO"
        self.go_start_time = rospy.Time.now().to_sec()

    # =====================================================================
    def go_mode(self):
        t = rospy.Time.now().to_sec() - self.go_start_time
        self.speed_pub.publish(Float64(self.ERPM_GO))
        self.show_marker(0.4, (0,0,1), f"GO {t:.1f}s")

        if t >= self.COMMIT_TIME:
            self.speed_pub.publish(Float64(self.ERPM_STOP))
            self.show_marker(None, (0,0,1), "DONE")


if __name__ == "__main__":
    rospy.init_node("roundabout")
    Mission5Roundabout()
    rospy.spin()
