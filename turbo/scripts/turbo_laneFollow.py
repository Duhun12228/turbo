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
        self.debug_publisher2 = rospy.Publisher('/debugging_image2', Image, queue_size=10)  # 차선 영역 오버레이 발행

        # --- White-only HSV thresholds ---
        self.white_lower  = np.array([0,   0, 200])
        self.white_upper  = np.array([180, 60, 255])

        self.src_points = np.float32([
            [200, 285],
            [440, 285],
            [-15, 480],
            [655, 480]
        ])
        self.dst_points = np.float32([
            [130,   0],
            [510,   0],
            [130, 480],
            [510, 480]
        ])
        self.matrix = cv.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inv_matrix = cv.getPerspectiveTransform(self.dst_points, self.src_points)

        # --- 이미지/조향 파라미터 ---
        self.img_width  = rospy.get_param('~img_width', 640)
        self.img_height = rospy.get_param('~img_height', 480)

        # PID 제어 파라미터
        self.kp = 0.01
        self.ki = 0.0001
        self.kd = 0.005

        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_steer_angle = 0.0  # smoothing 위한 이전 스티어링 값

        # 속도 설계 파라미터 (예시)
        self.base_speed = 1.0  # m/s
        self.turn_slowdown_k = 0.7
        self.v_min = 0.3
        self.v_max = 2.0

        # 서보 및 모터 변환 파라미터 (예시)
        self.steer_to_servo_gain = -1.2135
        self.steer_to_servo_offset = 0.5304
        self.speed_to_erpm_gain = 2000.0
        self.speed_to_erpm_offset = 0.0

    def image_cb(self, image_msg: CompressedImage):
        # 1) 압축 해제
        bgr = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg, 'bgr8')
        self.img_height, self.img_width = bgr.shape[:2]

        # 2) 흰색만 이진화 (원본 bgr 기준으로 유지)
        hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        white_mask = cv.inRange(hsv, self.white_lower, self.white_upper)
        white_mask = cv.medianBlur(white_mask, 5)
        white_mask = cv.morphologyEx(
            white_mask, cv.MORPH_CLOSE,
            cv.getStructuringElement(cv.MORPH_RECT, (5,5)), iterations=1
        )
        cv.imshow('white_mask',white_mask)
        
        # 3) 버드아이 변환
        warped = cv.warpPerspective(white_mask, self.matrix, (self.img_width, self.img_height))
        bgr_warped = cv.warpPerspective(bgr, self.matrix, (self.img_width, self.img_height))

        # 4) 슬라이딩 윈도우로 차선 검출
        left_fit, right_fit, left_lane_inds, right_lane_inds, out_img = self.sliding_window_polyfit(warped)

        # 5) 차선 중심과 차량 중심 계산
        lane_center = self.calculate_lane_center(left_fit, right_fit, warped.shape[0])
        image_center = self.img_width / 2
        error = image_center - lane_center

        # 6) PID 제어로 steering angle 계산
        steer_angle = self.pid_control(error)

        # 7) 속도 조절 (커브시 감속)
        speed = self.adjust_speed(steer_angle)

        # 8) 서보, 모터 명령으로 변환 후 발행
        servo_cmd = self.steer_to_servo_gain * steer_angle + self.steer_to_servo_offset
        motor_cmd = speed * self.speed_to_erpm_gain + self.speed_to_erpm_offset

        self.servo_pub.publish(Float64(servo_cmd))
        self.motor_pub.publish(Float64(motor_cmd))

        # 9) 디버깅 이미지 생성 및 발행
        debug_img1 = self.create_debug_image(out_img, left_lane_inds, right_lane_inds)
        debug_img_msg1 = self.cv_bridge.cv2_to_imgmsg(debug_img1, encoding="bgr8")
        self.debug_publisher1.publish(debug_img_msg1)

        debug_img2 = self.create_lane_overlay(bgr, left_fit, right_fit)
        debug_img_msg2 = self.cv_bridge.cv2_to_imgmsg(debug_img2, encoding="bgr8")
        self.debug_publisher2.publish(debug_img_msg2)

        # OpenCV 윈도우 띄워서 디버깅 (원하면 주석처리 가능)
        cv.imshow('Warped', warped)
        cv.imshow('Debug1', debug_img1)
        cv.imshow('Debug2 - Lane Overlay', debug_img2)
        cv.waitKey(1)

    def sliding_window_polyfit(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        midpoint = np.int32(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = np.int32(binary_warped.shape[0]//nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = 100
        minpix = 50

        left_lane_inds = []
        right_lane_inds = []

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # 윈도우 박스 그리기 (디버깅용)
            cv.rectangle(out_img, (win_xleft_low, win_y_low),
                         (win_xleft_high, win_y_high), (0,255,0), 2)
            cv.rectangle(out_img, (win_xright_low, win_y_low),
                         (win_xright_high, win_y_high), (0,255,0), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.array([0,0,0], dtype=np.float32)
        right_fit = np.array([0,0,0], dtype=np.float32)
        if len(leftx) > 0 and len(lefty) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 0 and len(righty) > 0:
            right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit, left_lane_inds, right_lane_inds, out_img

    def calculate_lane_center(self, left_fit, right_fit, y_eval):
        if left_fit is None or right_fit is None:
            return self.img_width / 2  # fallback

        left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        lane_center = (left_x + right_x) / 2
        return lane_center

    def pid_control(self, error):
        self.integral += error
        derivative = error - self.prev_error
        steer_angle = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Steering angle smoothing (저번 값과 적당히 섞기)
        alpha = 0.7
        steer_angle = alpha * steer_angle + (1 - alpha) * self.prev_steer_angle
        self.prev_steer_angle = steer_angle
        self.prev_error = error
        return steer_angle

    def adjust_speed(self, steer_angle):
        speed = self.base_speed * (1 - self.turn_slowdown_k * abs(steer_angle))
        speed = max(self.v_min, min(self.v_max, speed))
        return speed

    def create_debug_image(self, out_img, left_lane_inds, right_lane_inds):
        # 차선 픽셀 빨강, 파랑 표시
        out_img = out_img.copy()
        # 픽셀 좌표 가져오기
        # 여기서는 원본 픽셀 좌표가 없으므로 단순히 윈도우 영역만 그림(기존 out_img 사용)
        # 필요시 개선 가능
        return out_img

    def create_lane_overlay(self, orig_img, left_fit, right_fit):
        # 원본 영상에 차선 영역 반투명 오버레이 생성
        overlay = orig_img.copy()
        ploty = np.linspace(0, orig_img.shape[0]-1, orig_img.shape[0])
        color_warp = np.zeros_like(orig_img)

        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # 차선 영역 색칠
            cv.fillPoly(color_warp, np.int32([pts]), (255, 150, 50))

            # 버드아이 -> 원본 이미지 변환 (투시 변환 역행렬 사용)
            newwarp = cv.warpPerspective(color_warp, self.inv_matrix, (orig_img.shape[1], orig_img.shape[0]))

            # 반투명 오버레이
            cv.addWeighted(newwarp, 0.5, overlay, 1, 0, overlay)

        return overlay


if __name__ == '__main__':
    try:
        lf = LaneFollow()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
