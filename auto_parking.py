#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
from ackermann_msgs.msg import AckermannDriveStamped


class AutoParkingDirect:
    """
    Simple 3-step auto parking maneuver using the SAME drive logic style
    as lanefollow and parking_finder:

      - AckermannDriveStamped
      - speed in m/s
      - steering in radians
      - publishes to /high_level/ackermann_cmd_mux/input/nav_0
    """

    def __init__(self):
        # Do NOT call rospy.init_node() here.
        # It will be called by the main script or by the node that imports this.
        self.ack_pub = rospy.Publisher(
            "/high_level/ackermann_cmd_mux/input/nav_0",
            AckermannDriveStamped,
            queue_size=10
        )

        # ---- Parameters (tune on track as needed) ----
        # Speeds (m/s)
        self.reverse_speed = rospy.get_param("~reverse_speed", -0.25)   # slow reverse
        self.forward_speed = rospy.get_param("~forward_speed", 0.20)    # slow forward

        # Steering angles (radians)
        self.steer_angle = rospy.get_param("~steer_angle", math.radians(40.0))
        self.counter_steer_angle = rospy.get_param("~counter_steer_angle", math.radians(-40.0))

        # Durations (seconds)
        self.first_reverse_time = rospy.get_param("~first_reverse_time", 2.0)
        self.counter_reverse_time = rospy.get_param("~counter_reverse_time", 2.0)
        self.final_forward_time = rospy.get_param("~final_forward_time", 0.6)

        # Control rate
        self.control_rate = rospy.get_param("~control_rate", 20.0)

    # ============================================================
    # Low-level drive helpers
    # ============================================================
    def send_cmd(self, speed, steering):
        """
        Send a single Ackermann command.
        speed    [m/s]
        steering [rad]
        """
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.ack_pub.publish(msg)

    def timed_move(self, speed, steering, duration):
        """
        Hold a given speed & steering for 'duration' seconds.
        Uses the same idea as lane-follow: continuous streaming of
        AckermannDriveStamped at a fixed rate.
        """
        rate = rospy.Rate(self.control_rate)
        end_time = rospy.Time.now() + rospy.Duration.from_sec(duration)

        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self.send_cmd(speed, steering)
            rate.sleep()

        # brief brake
        self.send_cmd(0.0, 0.0)
        rospy.sleep(0.2)

    # ============================================================
    # Main parking routine
    # ============================================================
    def run_parking(self):
        """
        Execute a 3-step auto parking sequence:

          1) Reverse with steering into the spot
          2) Reverse with opposite steering to align in the box
          3) Forward straight to settle / center
        """
        rospy.loginfo("=== AUTO PARKING DIRECT START ===")

        rospy.loginfo("STEP 1 → Reverse with steering into spot")
        self.timed_move(self.reverse_speed, self.steer_angle, self.first_reverse_time)

        rospy.loginfo("STEP 2 → Reverse with counter-steer to align")
        self.timed_move(self.reverse_speed, self.counter_steer_angle, self.counter_reverse_time)

        rospy.loginfo("STEP 3 → Forward straight to settle")
        self.timed_move(self.forward_speed, 0.0, self.final_forward_time)

        rospy.loginfo("=== AUTO PARKING COMPLETE ===")
        self.send_cmd(0.0, 0.0)
        rospy.sleep(0.3)


# ============================================================
# Standalone usage (for testing only)
# ============================================================
if __name__ == "__main__":
    rospy.init_node("auto_parking_direct")
    ap = AutoParkingDirect()
    ap.run_parking()
