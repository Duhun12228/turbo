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


class ConeFollowerEclipse:
    def __init__(self):
        rospy.loginfo("=== CONE FOLLOWER ECLIPSE (LiDAR + Pure Pursuit + VESC) ===")

        # ---------- ROS I/O ----------
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        self.speed_pub = rospy.Publisher(
            "/commands/motor/speed", Float64, queue_size=1
        )
        self.steer_pub = rospy.Publisher(
            "/commands/servo/position", Float64, queue_size=1
        )

        # RViz ko‘rsatkichlari
        self.raw_pub    = rospy.Publisher("/cone_raw_points",   Marker, queue_size=1)
        self.center_pub = rospy.Publisher("/cone_centers",      Marker, queue_size=1)
        self.mid_pub    = rospy.Publisher("/cone_midpoints",    Marker, queue_size=1)
        self.target_pub = rospy.Publisher("/cone_target_point", Marker, queue_size=1)

        # ---------- ROI (old tomondagi corridor)  ----------
        # LiDAR tunnel kodidagi kabi teskari o‘rnatilgan deb faraz qilamiz.
        self.ROI_X_MIN = 0.20
        self.ROI_X_MAX = 2.00
        self.ROI_Y_MIN = -0.70
        self.ROI_Y_MAX =  0.70

        # ---------- DBSCAN parametrlari ----------
        self.DBSCAN_EPS         = 0.18   # cluster radius (m)  <<< TUNABLE
        self.DBSCAN_MIN_SAMPLES = 3

        # ---------- Mashina parametrlari ----------
        self.WHEELBASE = 0.28
        self.MAX_STEER = math.radians(30.0)

        # Lane eni diapazoni (chap-o‘ng orasidagi masofa)
        self.LANE_MIN = 0.40            # <<< TUNABLE: minimal corridor eni
        self.LANE_MAX = 1.00            # <<< TUNABLE: maksimal corridor eni

        # Pure Pursuit
        self.LOOKAHEAD = 0.70           # maqsad nuqta masofasi (m)  <<< TUNABLE

        # Tezlik (m/s ga o‘xshash)
        self.BASE_SPEED   = 0.10        # to‘g‘ri joyda shu atrofida
        self.MIN_SPEED    = 0.04        # juda sekin
        self.SLOWDOWN_GAIN = 0.05       # burilishda shuncha kamaytir

        # Faqat bitta qatordagi kon bo‘lsa (fallback)
        self.ONE_SIDE_OFFSET = 0.30     # qarshi tomonga qancha siljish (m)
        self.ONE_SIDE_FWD    = 0.50     # oldinga offset (m)

        # VESC mapping (tunnel kodiga mos)
        self.ERPM_BASE = 2000.0         # BASE_SPEED uchun ERPM

        # Ichki holat
        self.last_stamp = None

    # =========================================================
    # LiDAR callback
    # =========================================================
    def lidar_cb(self, scan):
        # ----- vaqt -----
        if scan.header.stamp.to_sec() > 0:
            now = scan.header.stamp.to_sec()
        else:
            now = rospy.get_time()

        if self.last_stamp is None:
            self.last_stamp = now
            return
        dt = now - self.last_stamp
        self.last_stamp = now

        # ----- ranges & angles -----
        ranges = np.array(scan.ranges, dtype=float)
        angles_orig = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles_orig = angles_orig[valid]

        if len(ranges) == 0:
            self.publish_drive(0.0, self.MIN_SPEED)
            self.publish_markers([], [], [], [], None)
            return

        # LiDAR teskari bo‘lgani uchun tunnel kodidagi kabi +pi
        angles = angles_orig + math.pi

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        # ----- ROI -----
        mask_roi = (
            (xs > self.ROI_X_MIN) & (xs < self.ROI_X_MAX) &
            (ys > self.ROI_Y_MIN) & (ys < self.ROI_Y_MAX)
        )
        xs_roi = xs[mask_roi]
        ys_roi = ys[mask_roi]

        cone_centers = []
        midpoints    = []
        target       = None

        if len(xs_roi) == 0:
            # Hech narsa ko‘rinmasa – sekin to‘g‘ri
            self.publish_drive(0.0, self.MIN_SPEED)
            self.publish_markers(xs_roi, ys_roi, [], [], None)
            rospy.loginfo("[CONE] ROI empty → straight, slow")
            return

        pts = np.vstack((xs_roi, ys_roi)).T

        # ----- DBSCAN orqali konlarni topamiz -----
        db = DBSCAN(eps=self.DBSCAN_EPS,
                    min_samples=self.DBSCAN_MIN_SAMPLES).fit(pts)
        labels = db.labels_

        unique = [l for l in set(labels) if l != -1]
        for lab in unique:
            cpts = pts[labels == lab]
            cx = float(np.mean(cpts[:, 0]))
            cy = float(np.mean(cpts[:, 1]))
            cone_centers.append((cx, cy))

        if len(cone_centers) == 0:
            self.publish_drive(0.0, self.MIN_SPEED)
            self.publish_markers(xs_roi, ys_roi, [], [], None)
            rospy.loginfo("[CONE] No clusters → slow straight")
            return

        # ----- targetni hisoblash -----
        target, midpoints = self.compute_target(cone_centers)

        if target is None:
            # target chiqmasa ham to‘g‘ri sekin yuramiz
            self.publish_drive(0.0, self.MIN_SPEED)
            self.publish_markers(xs_roi, ys_roi, cone_centers, midpoints, None)
            rospy.loginfo("[CONE] No valid target → safe straight")
            return

        tx, ty = target
        steer, speed = self.pure_pursuit(tx, ty)

        self.publish_drive(steer, speed)
        self.publish_markers(xs_roi, ys_roi, cone_centers, midpoints, target)

        rospy.loginfo(
            "[CONE] TGT=(%.2f,%.2f)  steer=%.1f°  speed=%.3f"
            % (tx, ty, math.degrees(steer), speed)
        )

    # =========================================================
    # Target hisoblash (eclipse / curved cones uchun)
    # =========================================================
    def compute_target(self, centers):
        """
        centers: [(x,y), ...] – DBSCAN centroids
        return: (target_x, target_y), midpoints_list
        """

        centers_np = np.array(centers, dtype=float)

        # Y bo‘yicha chap / o‘ng
        left  = centers_np[centers_np[:, 1] > 0.0]
        right = centers_np[centers_np[:, 1] < 0.0]

        midpoints = []

        # ---- 1) Ikkala tomonda ham cones bor → centerline ----
        if left.size > 0 and right.size > 0:
            # x bo‘yicha sort, lekin juftlashni "eng yaqin x" bilan qilamiz
            left_idx_sorted  = np.argsort(left[:, 0])
            right_idx_sorted = np.argsort(right[:, 0])

            used_right = set()

            for li in left_idx_sorted:
                lx, ly = left[li]
                best_j  = None
                best_dx = 1e9

                for rj in right_idx_sorted:
                    if rj in used_right:
                        continue
                    rx, ry = right[rj]

                    lane = math.hypot(lx - rx, ly - ry)
                    if lane < self.LANE_MIN or lane > self.LANE_MAX:
                        continue

                    dx = abs(lx - rx)
                    if dx < best_dx:
                        best_dx = dx
                        best_j  = rj

                if best_j is not None:
                    used_right.add(best_j)
                    rx, ry = right[best_j]
                    mx = 0.5 * (lx + rx)
                    my = 0.5 * (ly + ry)
                    if mx > 0.0:
                        midpoints.append((mx, my))

            if len(midpoints) > 0:
                # lookaheadga yaqin midpointni tanlaymiz
                dists = [math.hypot(mx, my) for (mx, my) in midpoints]
                idx = min(
                    range(len(midpoints)),
                    key=lambda k: abs(dists[k] - self.LOOKAHEAD)
                )
                return midpoints[idx], midpoints

        # ---- 2) Faqat bitta tomonda cones (fallback) ----
        # Bu holat eclipse oxiridagi yoki boshidagi yarim corridor bo‘lishi mumkin
        if left.size > 0 and right.size == 0:
            # eng yaqin oldingi cone
            li = np.argmin(left[:, 0])
            lx, ly = left[li]
            tx = lx + self.ONE_SIDE_FWD
            ty = ly - self.ONE_SIDE_OFFSET    # qarshi tomonga
            return (tx, ty), midpoints

        if right.size > 0 and left.size == 0:
            ri = np.argmin(right[:, 0])
            rx, ry = right[ri]
            tx = rx + self.ONE_SIDE_FWD
            ty = ry + self.ONE_SIDE_OFFSET
            return (tx, ty), midpoints

        # umuman taraflarni ajrata olmadik
        return None, midpoints

    # =========================================================
    # Pure Pursuit
    # =========================================================
    def pure_pursuit(self, tx, ty):
        Ld = math.hypot(tx, ty)
        if Ld < 0.05:
            return 0.0, self.MIN_SPEED

        alpha = math.atan2(ty, tx)
        delta = math.atan2(2.0 * self.WHEELBASE * math.sin(alpha), Ld)
        # cheklash
        if delta > self.MAX_STEER:
            delta = self.MAX_STEER
        if delta < -self.MAX_STEER:
            delta = -self.MAX_STEER

        # Burilish qanchalik katta bo‘lsa – shunchalik sekin
        turn_ratio = abs(delta) / self.MAX_STEER
        speed = self.BASE_SPEED - self.SLOWDOWN_GAIN * turn_ratio
        if speed < self.MIN_SPEED:
            speed = self.MIN_SPEED
        if speed > self.BASE_SPEED:
            speed = self.BASE_SPEED

        return delta, speed

    # =========================================================
    # VESC drive publish (tunnel kodi bilan bir xil mapping)
    # =========================================================
    def publish_drive(self, steer, speed_mps_like):
        # steer [rad] → servo [0.0–1.0]
        servo = 0.5 - (steer / self.MAX_STEER) * 0.5
        if servo < 0.0:
            servo = 0.0
        if servo > 1.0:
            servo = 1.0

        # speed → ERPM
        ratio = 0.0
        if self.BASE_SPEED > 1e-4:
            ratio = speed_mps_like / self.BASE_SPEED
        if ratio < 0.0:
            ratio = 0.0
        erpm = self.ERPM_BASE * ratio

        self.steer_pub.publish(Float64(servo))
        self.speed_pub.publish(Float64(erpm))

    # =========================================================
    # RViz vizualizatsiya
    # =========================================================
    def publish_markers(self, xs_roi, ys_roi, centers, mids, target):
        # Raw points (gray)
        raw = Marker()
        raw.header.frame_id = "laser"
        raw.type = Marker.POINTS
        raw.action = Marker.ADD
        raw.scale.x = 0.03
        raw.scale.y = 0.03
        raw.color.r = 0.5
        raw.color.g = 0.5
        raw.color.b = 0.5
        raw.color.a = 0.7
        for x, y in zip(xs_roi, ys_roi):
            raw.points.append(Point(x=x, y=y, z=0.0))
        self.raw_pub.publish(raw)

        # Cone centers (yellow)
        cent = Marker()
        cent.header.frame_id = "laser"
        cent.type = Marker.POINTS
        cent.action = Marker.ADD
        cent.scale.x = 0.07
        cent.scale.y = 0.07
        cent.color.r = 1.0
        cent.color.g = 1.0
        cent.color.b = 0.0
        cent.color.a = 1.0
        for cx, cy in centers:
            cent.points.append(Point(x=cx, y=cy, z=0.0))
        self.center_pub.publish(cent)

        # Midpoints (blue)
        mid = Marker()
        mid.header.frame_id = "laser"
        mid.type = Marker.POINTS
        mid.action = Marker.ADD
        mid.scale.x = 0.08
        mid.scale.y = 0.08
        mid.color.b = 1.0
        mid.color.a = 1.0
        for mx, my in mids:
            mid.points.append(Point(x=mx, y=my, z=0.0))
        self.mid_pub.publish(mid)

        # Target point (green)
        tgt = Marker()
        tgt.header.frame_id = "laser"
        tgt.type = Marker.SPHERE
        tgt.action = Marker.ADD
        tgt.scale.x = 0.15
        tgt.scale.y = 0.15
        tgt.scale.z = 0.15
        tgt.color.g = 1.0
        tgt.color.a = 1.0
        if target is not None:
            tx, ty = target
            tgt.pose.position.x = tx
            tgt.pose.position.y = ty
        else:
            tgt.color.a = 0.0
        self.target_pub.publish(tgt)


if __name__ == "__main__":
    rospy.init_node("cone_follower_eclipse")
    node = ConeFollowerEclipse()
    rospy.spin()
