#!/usr/bin/env python3
import rospy
import numpy as np
import math
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker


class BarrierMission:
    def __init__(self):
        rospy.loginfo("=== Mission 7: Barrier Gate (Corrected Front = 180°) ===")

        # Distances
        self.STOP_DISTANCE = 0.60       # stop when barrier is close
        self.CLEAR_DISTANCE = 1.20      # barrier considered open

        # Delay after open
        self.DELAY_AFTER_OPEN = 5.0

        # Straight steering (center)
        self.CENTER = 0.5

        # State
        self.waiting = False
        self.open_time = None

        # ROS I/O
        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.speed_pub  = rospy.Publisher("/commands/motor/speed", Float64, queue_size=1)
        self.steer_pub  = rospy.Publisher("/commands/servo/position", Float64, queue_size=1)

        # RViz visualization
        self.marker_pub = rospy.Publisher("/mission7_marker", Marker, queue_size=1)
        self.text_pub   = rospy.Publisher("/mission7_text", Marker, queue_size=1)

        rospy.Timer(rospy.Duration(0.05), self.update)


    # -------------------------------------------------------
    # LIDAR
    # -------------------------------------------------------
    def scan_callback(self, scan):
        self.scan = scan

    def get_front_dist(self):
        """
        Reads LiDAR around 180 degrees because the LiDAR is mounted backwards.
        We treat ±10° around 180° as FRONT OF CAR.
        """
        if not hasattr(self, "scan"):
            return None

        ranges = np.array(self.scan.ranges, dtype=float)
        angles = self.scan.angle_min + np.arange(len(ranges)) * self.scan.angle_increment

        # Car's front direction = 180 degrees (pi radians)
        front_center = math.pi          # 180°
        half_width   = math.radians(20) # ±10°

        # Angular difference with wrap-around: shortest difference between angles
        diff = np.arctan2(np.sin(angles - front_center),
                          np.cos(angles - front_center))

        mask = np.abs(diff) < half_width

        sector = ranges[mask]
        sector = sector[np.isfinite(sector)]

        if len(sector) == 0:
            return 5.0

        return float(np.mean(sector))


    # -------------------------------------------------------
    # RVIZ visualization
    # -------------------------------------------------------
    def show_marker(self, x, color):
        m = Marker()
        m.header.frame_id = "laser"
        m.type = Marker.SPHERE
        m.scale.x = m.scale.y = m.scale.z = 0.15
        m.color.r, m.color.g, m.color.b = color
        m.color.a = 1.0
        m.pose.position.x = -x
        m.pose.position.y = 0
        self.marker_pub.publish(m)

    def show_text(self, text):
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


    # -------------------------------------------------------
    # MAIN MISSION LOGIC
    # -------------------------------------------------------
    def update(self, event):
        dist = self.get_front_dist()
        if dist is None:
            return

        # Always keep steering centered for this mission
        self.steer_pub.publish(Float64(self.CENTER))

        # Default visualization = green sphere
        self.show_marker(min(dist, 2.0), (0,1,0))
        self.show_text("APPROACH")

        # ==========================================
        # 1. APPROACH → Drive slowly until barrier close
        # ==========================================
        if not self.waiting:
            if dist > self.STOP_DISTANCE:
                self.speed_pub.publish(Float64(1200))   # slow forward
            else:
                rospy.loginfo("[Mission7] Barrier detected → STOP & WAIT")
                self.speed_pub.publish(Float64(0))
                self.waiting = True
            return

        # ==========================================
        # 2. WAITING FOR BARRIER TO OPEN
        # ==========================================
        if self.open_time is None:
            self.show_text("WAITING")
            self.show_marker(dist, (1,0,0))
            self.speed_pub.publish(Float64(0))

            if dist > self.CLEAR_DISTANCE:
                rospy.loginfo("[Mission7] Barrier opened!")
                self.open_time = rospy.Time.now().to_sec()
            return

        # ==========================================
        # 3. DELAY AFTER OPEN (2 seconds)
        # ==========================================
        if rospy.Time.now().to_sec() - self.open_time < self.DELAY_AFTER_OPEN:
            self.show_text("DELAY (2s)")
            self.speed_pub.publish(Float64(0))
            return

        # ==========================================
        # 4. GO!
        # ==========================================
        rospy.loginfo("[Mission7] GO!")
        self.show_text("GO!")
        self.speed_pub.publish(Float64(1200))



if __name__ == "__main__":
    rospy.init_node("barrier_mission")
    BarrierMission()
    rospy.spin()

