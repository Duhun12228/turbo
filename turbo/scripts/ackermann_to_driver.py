#!/usr/bin/env python3
import rospy
import cv2 as cv
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64

class LaneFollow:
    def __init__(self):
        rospy.init_node('lane_follow')
        self.cv_bridge = CvBridge()

        # --- I/O ---
        self.image_sub = rospy.Subscriber('/usb_cam/image_rect_color/compressed',
                                          CompressedImage, self.image_cb, queue_size=1)
        self.motor_pub = rospy.Publisher('/commands/motor/speed', Float64, queue_size=10)
        self.servo_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=10)
        self.debug_publisher1 = rospy.Publisher('/debugging_image1', Image, queue_size=10)

        # --- White-only HSV thresholds (평탄화 후 기준) ---
        self.white_lower  = np.array([0,   0, 200])
        self.white_upper  = np.array([179, 50, 255])

        # --- Perspective points (원본 이미지 좌표 기준) ---
        self.src_points = np.float32([
            [220, 300],  # 좌상
            [420, 300],  # 우상
            [0,   480],  # 좌하
            [640, 480]   # 우하
        ])
        self.dst_points = np.float32([
            [150,   0],
            [490,   0],
            [150, 480],
            [490, 480]
        ])
        self.matrix = cv.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inv_matrix = cv.getPerspectiveTransform(self.dst_points, self.src_points)

        # --- 이미지 크기/조향 계산 파라미터 ---
        self.img_width  = rospy.get_param('~img_width', 640)
        self.img_height = rospy.get_param('~img_height', 480)
        self.lookahead_px = rospy.get_param('~lookahead_px', 300)  # atan2 분모

        # --- 속도(m/s) 계산 파라미터 (이 값은 erpm 변환 전 속도 설계용) ---
        self.base_speed_mps = rospy.get_param('~base_speed_mps', 1.0)
        self.turn_slowdown_k = rospy.get_param('~turn_slowdown_k', 0.7)
        self.v_min_mps = rospy.get_param('~v_min_mps', 0.3)
        self.v_max_mps = rospy.get_param('~v_max_mps', 2.0)

        # --- 슬라이딩 윈도우 파라미터 ---
        self.n_windows = rospy.get_param('~n_windows', 9)
        self.margin    = rospy.get_param('~window_margin', 50)
        self.minpix    = rospy.get_param('~window_minpix', 50)

        # --- ⚙️ 최종 퍼블리시 전에 적용할 변환 (AckermannToVESC와 동일 키) ---
        self.speed_to_erpm_gain     = rospy.get_param("~speed_to_erpm_gain", 2000.0)
        self.speed_to_erpm_offset   = rospy.get_param("~speed_to_erpm_offset", 0.0)
        self.steer_to_servo_gain    = rospy.get_param("~steering_angle_to_servo_gain", -1.2135)
        self.steer_to_servo_offset  = rospy.get_param("~steering_angle_to_servo_offset", 0.5304)

    # ----------------- CALLBACK -----------------
    def image_cb(self, image_msg: CompressedImage):
        # 1) 압축 해제
        bgr = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg, 'bgr8')
        self.img_height, self.img_width = bgr.shape[:2]

        # 2) 전 프레임 평탄화 (illumination flattening on V)
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        bg = cv.GaussianBlur(v, (0,0), sigmaX=21, sigmaY=21)
        v_flat = cv.divide(v, np.maximum(bg, 1), scale=255)
        v_flat = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(v_flat)
        hsv_flat = cv.merge([h, s, v_flat])
        bgr_flat = cv.cvtColor(hsv_flat, cv.COLOR_HSV2BGR)

        # 3) 흰색만 이진화
        hsv2 = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        white_mask = cv.inRange(hsv2, self.white_lower, self.white_upper)
        white_mask = cv.medianBlur(white_mask, 5)
        white_mask = cv.morphologyEx(
            white_mask, cv.MORPH_CLOSE,
            cv.getStructuringElement(cv.MORPH_RECT, (5,5)), iterations=1
        )

        # 4) 버드아이 변환
        warped = cv.warpPerspective(white_mask, self.matrix, (self.img_width, self.img_height))

        # 5) center 계산
        center_x, debug_viz = self.calculate_center(warped)

        # 6) 스티어링/속도 계산 → ⚙️ 변환 후 기존 토픽으로 퍼블리시
        self.publish_with_conversion(center_x, debug_viz)

        # 7) 디버그 퍼블리시/창
        try:
            self.debug_publisher1.publish(self.cv_bridge.cv2_to_imgmsg(debug_viz, encoding='bgr8'))
        except:
            pass
        cv.imshow('flat_bgr', bgr_flat)
        cv.imshow('white_mask', white_mask)
        cv.imshow('warped_bin', warped)
        cv.waitKey(1)

    # ----------------- CENTER LINE -----------------
    def calculate_center(self, warped_bin: np.ndarray):
        h, w = warped_bin.shape[:2]
        out_viz = cv.cvtColor(warped_bin, cv.COLOR_GRAY2BGR)

        # 하단 히스토그램으로 좌/우 시작점
        histogram = np.sum(warped_bin[h//2:,:], axis=0)
        midpoint = w // 2
        leftx_base  = int(np.argmax(histogram[:midpoint]))
        rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)

        nonzero = warped_bin.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        window_height = int(h / self.n_windows)
        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(self.n_windows):
            win_y_low  = h - (window + 1) * window_height
            win_y_high = h - window * window_height
            win_xleft_low  = leftx_current  - self.margin
            win_xleft_high = leftx_current  + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high= rightx_current + self.margin

            cv.rectangle(out_viz, (win_xleft_low, win_y_low),
                         (win_xleft_high, win_y_high), (0,255,0), 2)
            cv.rectangle(out_viz, (win_xright_low, win_y_low),
                         (win_xright_high, win_y_high), (255,0,0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds  = np.concatenate(left_lane_inds)  if len(left_lane_inds)  else []
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else []

        leftx  = nonzerox[left_lane_inds]  if len(left_lane_inds)  else np.array([])
        lefty  = nonzeroy[left_lane_inds]  if len(left_lane_inds)  else np.array([])
        rightx = nonzerox[right_lane_inds] if len(right_lane_inds) else np.array([])
        righty = nonzeroy[right_lane_inds] if len(right_lane_inds) else np.array([])

        left_fit = np.polyfit(lefty, leftx, 2) if (len(leftx) > 50 and len(lefty) > 50) else None
        right_fit = np.polyfit(righty, rightx, 2) if (len(rightx) > 50 and len(righty) > 50) else None

        y_eval = h - 1
        had_left  = left_fit  is not None
        had_right = right_fit is not None

        if had_left:
            left_x_eval = int(left_fit[0]*y_eval*y_eval + left_fit[1]*y_eval + left_fit[2])
            cv.circle(out_viz, (left_x_eval, y_eval), 6, (0,255,0), -1)
        else:
            left_x_eval = None

        if had_right:
            right_x_eval = int(right_fit[0]*y_eval*y_eval + right_fit[1]*y_eval + right_fit[2])
            cv.circle(out_viz, (right_x_eval, y_eval), 6, (255,0,0), -1)
        else:
            right_x_eval = None

        if had_left and had_right:
            center_x = int((left_x_eval + right_x_eval) // 2)
        elif had_left:
            lane_half = 170  # px 추정치
            center_x = int(left_x_eval + lane_half)
        elif had_right:
            lane_half = 170
            center_x = int(right_x_eval - lane_half)
        else:
            center_x = w // 2

        cv.line(out_viz, (center_x, h-30), (center_x, h), (0,255,255), 3)
        cv.line(out_viz, (w//2, 0), (w//2, h), (0,255,255), 1)
        return center_x, out_viz

    # ----------------- 최종 퍼블리시(변환 포함) -----------------
    def publish_with_conversion(self, center_x: int, debug_viz):
        """
        1) 픽셀 기반 steering angle(rad) 계산
        2) m/s 속도 계산
        3) ⚙️ AckermannToVESC와 같은 변환식을 적용해
           /commands/motor/speed(ERPM), /commands/servo/position(servo_pos)로 퍼블리시
        """
        img_center = self.img_width // 2
        dx = float(center_x - img_center)

        # (a) 스티어링 각 (rad): 좌(+), 우(-)
        steering_angle = math.atan2(dx, float(self.lookahead_px))

        # (b) 속도(m/s): 회전이 클수록 감속
        slow_ratio = 1.0 - self.turn_slowdown_k * min(abs(steering_angle)/0.6, 1.0)
        speed_mps = self.base_speed_mps * slow_ratio
        speed_mps = float(np.clip(speed_mps, self.v_min_mps, self.v_max_mps))

        # (c) ⚙️ 변환: speed→ERPM, steer→servo_pos
        erpm = self.speed_to_erpm_gain * speed_mps + self.speed_to_erpm_offset
        servo_pos = self.steer_to_servo_gain * steering_angle + self.steer_to_servo_offset

        # (d) 퍼블리시 (Float64)
        self.motor_pub.publish(Float64(data=erpm))
        self.servo_pub.publish(Float64(data=servo_pos))

        # 디버그 오버레이
        info = (f"dx={dx:.1f}px  steer={math.degrees(steering_angle):.2f}deg  "
                f"v={speed_mps:.2f}m/s  ERPM={erpm:.0f}  SERVO={servo_pos:.3f}")
        cv.putText(debug_viz, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv.LINE_AA)

# ----------------- MAIN -----------------
if __name__ == '__main__':
    try:
        lf = LaneFollow()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
