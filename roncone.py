#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import math
import numpy as np

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sklearn.cluster import DBSCAN
from ackermann_msgs.msg import AckermannDriveStamped
import time

class ConeFollowerEclipse:
    def __init__(self):
        rospy.init_node("cone_follower_eclipse")
        rospy.loginfo("=== CONE FOLLOWER ECLIPSE (Optimized Sharp Turns) ===")

        # ---------- ROS I/O ----------
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)
        self.ack_pub_1 = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=10)

        self.mission_sub = rospy.Subscriber('/mission_num',Float64,self.mission_cb,queue_size=10)
        # RViz markers
        self.mission_pub = rospy.Publisher('/mission_num', Float64, queue_size=10)

        self.raw_pub    = rospy.Publisher("/cone_raw_points",   Marker, queue_size=1)
        self.center_pub = rospy.Publisher("/cone_centers",      Marker, queue_size=1)
        self.mid_pub    = rospy.Publisher("/cone_midpoints",    Marker, queue_size=1)
        self.target_pub = rospy.Publisher("/cone_target_point", Marker, queue_size=1)
        self.start_time = 0
        
        # ---------- ROI ----------
        self.ROI_X_MIN = 0.
        self.ROI_X_MAX = 1.0
        self.ROI_Y_MIN = -0.20
        self.ROI_Y_MAX =  0.20

        # ---------- DBSCAN ----------
        self.DBSCAN_EPS = 0.16
        self.DBSCAN_MIN_SAMPLES = 3

        # ---------- Vehicle ----------
        self.WHEELBASE = 0.28

        # SHARPER TURNING: 40 degrees
        self.MAX_STEER = math.radians(40.0)

        self.LANE_MIN = 0.40
        self.LANE_MAX = 1.15
        self.PAIR_X_THRESH = 0.60

        # ---------- Lookahead ----------
        # SHARPER TURN VALUES
        self.LOOKAHEAD_MIN  = 0.40
        self.LOOKAHEAD_BASE = 0.55
        self.LOOKAHEAD_MAX  = 0.80

        # ---------- Speed ----------
        self.BASE_SPEED    = 0.10
        self.MIN_SPEED     = 0.04
        self.SLOWDOWN_GAIN = 0.05

        # ---------- One-side fallback ----------
        self.ONE_SIDE_OFFSET = 0.30
        self.ONE_SIDE_FWD    = 0.50

        # ---------- VESC ----------
        self.ERPM_BASE = 1600.0

        # ---------- Smoothing ----------
        self.STEER_SMOOTH_ALPHA = 0.35
        self.SPEED_SMOOTH_ALPHA = 0.45
        self.last_steer = 0.0
        self.last_speed = 0.0

        self.last_stamp = None

        self.cone_centers = []
        self.current_mission = 0.0
        self.state = None
        self.xs = None
        self.xy = None
        self.xs_roi = None
        self.xy_roi = None

    # =========================================================
    def mission_cb(self,msg):
        self.current_mission = msg.data

    def lidar_cb(self, scan):
        if scan.header.stamp.to_sec() > 0:
            now = scan.header.stamp.to_sec()
        else:
            now = rospy.get_time()

        if self.last_stamp is None:
            self.last_stamp = now
            return

        self.last_stamp = now

        ranges = np.array(scan.ranges, dtype=float)
        angles_orig = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles_orig = angles_orig[valid]

        if len(ranges) == 0:
            # self.publish_drive(0.0, self.MIN_SPEED)
            self.publish_markers([], [], [], [], None)
            return

        # LiDAR orientation (mounted backwards)
        angles = angles_orig + math.pi
        self.xs = ranges * np.cos(angles)
        self.ys = ranges * np.sin(angles)

        # ROI filtering
        mask_roi = (
            (self.xs > self.ROI_X_MIN) & (self.xs < self.ROI_X_MAX) &
            (self.ys > self.ROI_Y_MIN) & (self.ys < self.ROI_Y_MAX)
        )
        self.xs_roi = self.xs[mask_roi]
        self.ys_roi = self.ys[mask_roi]

        if len(self.xs_roi) == 0:
            self.publish_markers([], [], [], [], None)
            return

        pts = np.vstack((self.xs_roi, self.ys_roi)).T

        # ---------- DBSCAN clustering ----------
        db = DBSCAN(eps=self.DBSCAN_EPS, min_samples=self.DBSCAN_MIN_SAMPLES).fit(pts)
        labels = db.labels_

        self.cone_centers = []
        for lab in set(labels):
            if lab == -1:
                continue
            cpts = pts[labels == lab]
            self.cone_centers.append((float(np.mean(cpts[:, 0])), float(np.mean(cpts[:, 1]))))

    # =========================================================
    def smooth(self, prev, new, a):
        return a * prev + (1 - a) * new

    # =========================================================
    def compute_target(self, centers):
        centers_np = np.array(centers, dtype=float)

        left  = centers_np[centers_np[:, 1] > 0.0]
        right = centers_np[centers_np[:, 1] < 0.0]

        midpoints = []

        # BOTH SIDES visible
        if len(left) > 0 and len(right) > 0:
            left = left[np.argsort(left[:, 0])]
            right = right[np.argsort(right[:, 0])]

            used = set()

            for lx, ly in left:
                best_j = None
                best_cost = 1e9

                for j, (rx, ry) in enumerate(right):
                    if j in used:
                        continue

                    lane_w = math.hypot(lx - rx, ly - ry)
                    if not (self.LANE_MIN <= lane_w <= self.LANE_MAX):
                        continue

                    dx = abs(lx - rx)
                    if dx > self.PAIR_X_THRESH:
                        continue

                    preferred = 0.5 * (self.LANE_MIN + self.LANE_MAX)
                    cost = dx + abs(lane_w - preferred)

                    if cost < best_cost:
                        best_cost = cost
                        best_j = j

                if best_j is not None:
                    used.add(best_j)
                    rx, ry = right[best_j]
                    mx = 0.5 * (lx + rx)
                    my = 0.5 * (ly + ry)
                    if mx > 0:
                        midpoints.append((mx, my))

        # ONE SIDE fallback
        elif len(left) > 0 and len(right) ==0:
            idx = np.argmin(left[:, 0])
            lx, ly = left[idx]
            tx = lx + self.ONE_SIDE_FWD
            ty = -0.4
            return (tx, ty), midpoints

        elif len(right) > 0 and len(left) == 0:
            idx = np.argmin(right[:, 0])
            lx, ly = right[idx]

            tx = lx + self.ONE_SIDE_FWD
            ty = +0.4
            return (tx, ty), midpoints


        if len(midpoints) == 0:
            return None, midpoints

        # Sort midpoints by distance
        midpoints = sorted(midpoints, key=lambda p: math.hypot(p[0], p[1]))

        # Dynamic lookahead
        Ld = self.compute_dynamic_lookahead(midpoints)
        dists = [math.hypot(mx, my) for mx, my in midpoints]

        idx = min(range(len(midpoints)), key=lambda k: abs(dists[k] - Ld))

        return midpoints[idx], midpoints

    # =========================================================
    def compute_dynamic_lookahead(self, mids):
        if len(mids) < 3:
            return self.LOOKAHEAD_BASE

        p0 = np.array(mids[0])
        p1 = np.array(mids[len(mids)//2])
        p2 = np.array(mids[-1])

        ang1 = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
        ang2 = math.atan2(p2[1]-p1[1], p2[0]-p1[0])

        dtheta = abs((ang2 - ang1 + math.pi) % (2*math.pi) - math.pi)
        k = min(dtheta / (math.pi/2), 1.0)

        Ld = self.LOOKAHEAD_MAX - k*(self.LOOKAHEAD_MAX - self.LOOKAHEAD_MIN)
        return max(self.LOOKAHEAD_MIN, min(self.LOOKAHEAD_MAX, Ld))

    # =========================================================
    def pure_pursuit(self, tx, ty):
        Ld = math.hypot(tx, ty)
        if Ld < 0.05:
            return 0.0, self.MIN_SPEED

        alpha = math.atan2(ty, tx)
        delta = math.atan2(2.0 * self.WHEELBASE * math.sin(alpha), Ld)

        # clamp to 40° range
        delta = max(-self.MAX_STEER, min(self.MAX_STEER, delta))

        turn_ratio = abs(delta) / self.MAX_STEER
        speed = self.BASE_SPEED - self.SLOWDOWN_GAIN * turn_ratio
        speed = 0.2

        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'cone'
        msg.drive.steering_angle = delta
        msg.drive.speed = speed

        self.ack_pub_1.publish(msg)

        return delta, speed

    # =========================================================

    def publish_markers(self, xs_roi, ys_roi, centers, mids, target):
        raw = Marker()
        raw.header.frame_id = "laser"
        raw.type = Marker.POINTS
        raw.scale.x = raw.scale.y = 0.03
        raw.color.r = raw.color.g = raw.color.b = 0.5
        raw.color.a = 0.7
        for x, y in zip(xs_roi, ys_roi):
            raw.points.append(Point(x=float(-x), y=float(-y)))
        self.raw_pub.publish(raw)

        cent = Marker()
        cent.header.frame_id = "laser"
        cent.type = Marker.POINTS
        cent.scale.x = cent.scale.y = 0.07
        cent.color.r = 1.0
        cent.color.g = 1.0
        cent.color.b = 0.0
        cent.color.a = 1.0
        for (cx, cy) in centers:
            cent.points.append(Point(x=float(-cx), y=float(-cy)))
        self.center_pub.publish(cent)

        mid = Marker()
        mid.header.frame_id = "laser"
        mid.type = Marker.POINTS
        mid.scale.x = mid.scale.y = 0.08
        mid.color.b = 1.0
        mid.color.a = 1.0
        for (mx, my) in mids:
            mid.points.append(Point(x=float(-mx), y=float(-my)))
        self.mid_pub.publish(mid)

        tgt = Marker()
        tgt.header.frame_id = "laser"
        tgt.type = Marker.SPHERE
        tgt.scale.x = tgt.scale.y = tgt.scale.z = 0.15
        tgt.color.g = 1.0
        tgt.color.a = 1.0
        if target is not None:
            tx, ty = target
            tgt.pose.position.x = float(-tx)
            tgt.pose.position.y = float(-ty)
        else:
            tgt.color.a = 0.0
        self.target_pub.publish(tgt)


    def main(self):
        
        if self.state == None:
            if len(self.cone_centers) == 0:
                rospy.loginfo('Cone is not detected yet')
                return
            else:
                rospy.loginfo('Cone is detected.. following....')
                self.state = 'cone_following'
                target, midpoints = self.compute_target(self.cone_centers)
                tx, ty = target

                steer_raw, speed_raw = self.pure_pursuit(tx, ty)

                # Smoothing
                steer = self.smooth(self.last_steer, steer_raw, self.STEER_SMOOTH_ALPHA)
                speed = self.smooth(self.last_speed, speed_raw, self.SPEED_SMOOTH_ALPHA)
                self.last_steer = steer
                self.last_speed = speed

                # Publish
                # self.publish_drive(steer, speed) 
                self.publish_markers(self.xs_roi, self.ys_roi, self.cone_centers, midpoints, target)

        elif self.state == 'cone_following' and len(self.cone_centers) != 0:
            
                target, midpoints = self.compute_target(self.cone_centers)
                if target is None:
                    return
                tx, ty = target

                steer_raw, speed_raw = self.pure_pursuit(tx, ty)

                # Smoothing
                steer = self.smooth(self.last_steer, steer_raw, self.STEER_SMOOTH_ALPHA)
                speed = self.smooth(self.last_speed, speed_raw, self.SPEED_SMOOTH_ALPHA)
                self.last_steer = steer
                self.last_speed = speed

                # Publish
                self.publish_markers(self.xs_roi, self.ys_roi, self.cone_centers, midpoints, target)

        elif self.state == 'cone_following' and len(self.cone_centers) == 0:
                self.state = 'Done'
                rospy.loginfo('Cone follow is done!')

        elif self.state == 'Done':
                msg = Float64()
                msg.data = 4.
                self.current_mission = 0.
                self.mission_pub.publish(msg)
                rospy.loginfo('------mission 4------')
        

if __name__ =='__main__':
    try:
        cf = ConeFollowerEclipse()
        rate = rospy.Rate(25)  # 25Hz

        while not rospy.is_shutdown():
            if cf.current_mission == 3.0:
                if cf.start_time == 0:
                    cf.start_time = time.time()
                elif time.time() - cf.start_time > 5.0:
                   cf.main()
                   rate.sleep()

    except rospy.ROSInterruptException:
        pass