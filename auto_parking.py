#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
from ackermann_msgs.msg import AckermannDriveStamped


class AutoParkingDirect:
    """
    Updated for NEW COMPETITION PARKING ALGORITHM:
        1) sharp left turn from white line
        2) return back to white line position
        3) drive forward EXACT 105 cm
    """

    def __init__(self):
        self.ack_pub = rospy.Publisher(
            "/high_level/ackermann_cmd_mux/input/nav_0",
            AckermannDriveStamped,
            queue_size=10
        )

        # ---------- NEW PARAMETERS ----------
        # Speeds
        self.forward_speed = rospy.get_param("~forward_speed", 0.22)
        self.reverse_speed = rospy.get_param("~reverse_speed", -0.22)

        # Sharp turn steering angle
        self.sharp_left = rospy.get_param("~sharp_left", math.radians(43.0))
        self.straight = 0.0

        # Timings (tune on track)
        self.turn_into_time = rospy.get_param("~turn_into_time", 1.4)    # swing left
        self.return_time = rospy.get_param("~return_time", 1.4)          # return back
        self.forward_105_time = rospy.get_param("~forward_105_time", 1.0)  # forward 1.05m

        # publish rate
        self.control_rate = rospy.get_param("~control_rate", 20.0)

    # --- Low-level send ---
    def send_cmd(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.ack_pub.publish(msg)

    # --- Hold movement for duration ---
    def timed_move(self, speed, steering, duration):
        rate = rospy.Rate(self.control_rate)
        end_time = rospy.Time.now() + rospy.Duration.from_sec(duration)

        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self.send_cmd(speed, steering)
            rate.sleep()

        # brief brake
        self.send_cmd(0.0, 0.0)
        rospy.sleep(0.15)

    # ============== NEW ALGORITHM ==============
    def run_parking(self):
        rospy.loginfo("=== NEW OFFICIAL PARKING START ===")

        # --------------------------
        # Step 1: Sharp left turn
        # --------------------------
        rospy.loginfo("STEP 1: Sharp LEFT swing")
        self.timed_move(self.forward_speed, self.sharp_left, self.turn_into_time)

        # --------------------------
        # Step 2: Return to white line (reverse)
        # --------------------------
        rospy.loginfo("STEP 2: Returning back to white line")
        self.timed_move(self.reverse_speed, -self.sharp_left, self.return_time)

        # --------------------------
        # Step 3: Final 105 cm forward
        # --------------------------
        rospy.loginfo("STEP 3: Forward EXACT 105 cm")
        self.timed_move(self.forward_speed, 0.0, self.forward_105_time)

        rospy.loginfo("=== PARKING COMPLETE ===")
        self.send_cmd(0.0, 0.0)
        rospy.sleep(0.2)


if __name__ == "__main__":
    rospy.init_node("auto_parking_direct")
    ap = AutoParkingDirect()
    ap.run_parking()





def run_parking(self):
    rospy.loginfo("=== NEW OFFICIAL PARKING START ===")

    # --------------------------
    # STEP 1 → Sharp left turn into the parking area
    # --------------------------
    rospy.loginfo("STEP 1: Sharp LEFT swing into parking zone")
    self.timed_move(self.forward_speed, self.sharp_left, self.turn_into_time)

    # --------------------------
    # STEP 2 → STOP & stay inside for 3 seconds
    # --------------------------
    rospy.loginfo("STEP 2: STOPPING inside parking zone for REQUIRED 3 seconds")
    self.send_cmd(0.0, 0.0)   # full stop
    rospy.sleep(3.0)

    # --------------------------
    # STEP 3 → Return back to white line (reverse)
    # --------------------------
    rospy.loginfo("STEP 3: Returning back to white line")
    self.timed_move(self.reverse_speed, -self.sharp_left, self.return_time)

    # --------------------------
    # STEP 4 → Move forward EXACT 105 cm
    # --------------------------
    rospy.loginfo("STEP 4: Moving forward EXACT 105 cm")
    self.timed_move(self.forward_speed, 0.0, self.forward_105_time)

    # --------------------------
    # STEP 5 → Final stop / mission complete
    # --------------------------
    rospy.loginfo("=== PARKING MISSION COMPLETE ===")
    self.send_cmd(0.0, 0.0)
    rospy.sleep(0.2)




