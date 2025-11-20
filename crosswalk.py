import rospy
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
import time

class Crosswalk():
    def __init__(self):
        rospy.init_node('crosswalk')
        self.cv_bridge = CvBridge()
        self.image_sub = rospy.Subscriber(
            '/usb_cam/image_rect_color/compressed', CompressedImage, self.image_cb, queue_size=1
        )
        self.mission = rospy.Subscriber('/mission_num', Float64, self.mission_cb, queue_size=1)
        self.mission_pub = rospy.Publisher('/mission_num', Float64, queue_size=10)
        self.motor_pub = rospy.Publisher('/commands/motor/speed',  Float64, queue_size=10)
        self.ack_pub_1 = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=10)
        self.debug_publisher1 = rospy.Publisher('/cross_debug_roi', Image, queue_size=10)

        self.yellow_lower = np.array(rospy.get_param('~yellow_lower', [20, 40, 100]), dtype=np.uint8)
        self.yellow_upper = np.array(rospy.get_param('~yellow_lower',[38, 110, 255]), dtype=np.uint8)

        self.yellow_lower = np.array(rospy.get_param('~yellow_lower', [20, 40, 100]), dtype=np.uint8)
        self.yellow_upper = np.array(rospy.get_param('~yellow_lower',[38, 110, 255]), dtype=np.uint8)   

        self.bgr = None
        self.roi = None
        self.yellow_img = None
        self.filtered_img = None
        self.state = None
        self.start_time = 0

        self.current_mission = None

    def image_cb(self, msg):
        self.bgr = self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.roi = self.bgr[320:480, 0:640]

    def mission_cb(self,msg):
        self.current_mission = msg.data
    def yellow_color_filter_hsv(self,img):
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        white_hsv = cv.inRange(hsv,self.yellow_lower,self.yellow_upper)
        masked_img = cv.bitwise_and(img,img,mask=white_hsv)
        return masked_img
    
    def binary_filter(self,img):
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        _,binary =  cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
        return binary
    
    def Gaussian_filter(self,img):
        filtered_img = cv.GaussianBlur(img,(0,0),1)
        return filtered_img
    
    def detect_crosswalk(self,img, min_area=500,  # 최소 면적 (픽셀 수)
                               min_ratio=5.0): #img: 바이너리 이미지
        # 컨투어 찾기 (흰색(255)을 객체로 가정)
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            area = w * h

            # 너무 작은 건 무시
            if area < min_area:
                continue

            # 가로로 긴 성분인지 체크
            if h == 0:
                continue
            ratio = float(w) / float(h)

            if ratio >= min_ratio:
                # 조건 만족하는 '가로선 성분' 발견
                return "crosswalk"

        # 조건 만족 컨투어 없음
        return "none"

    def main(self):
        # 이미지 아직 안 들어왔으면 리턴
        if self.bgr is None:
            return
        self.debug_publisher1.publish(self.cv_bridge.cv2_to_imgmsg(self.roi, "bgr8"))

        g_filtered = self.Gaussian_filter(self.roi)
        self.yellow_img = self.yellow_color_filter_hsv(g_filtered)
        self.filtered_img = self.binary_filter(self.yellow_img)
        cv.imshow('filter img',self.filtered_img)
        cv.imshow('ori img',g_filtered)
        result = self.detect_crosswalk(self.filtered_img)

        now = time.time()

        # 1) 아직 아무 일도 안 일어났을 때: crosswalk 처음 감지되면 타이머 시작
        if self.state is None:
            if result == 'crosswalk':
                rospy.loginfo('crosswalk 찾기 완료!!')
                msg = AckermannDriveStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = 'Cross_walk'
                msg.drive.steering_angle = 0.0
                msg.drive.speed = 0.0

                self.ack_pub_1.publish(msg)
                self.start_time = now
                self.state = 'crosswalk'
                rospy.loginfo(f'정지 시작시간:{self.start_time}')
            else:
                rospy.loginfo('crosswalk 찾기 중...')

        # 2) 이미 crosswalk 상태일 때: 9초 동안은 계속 stop 발행
        elif self.state == 'crosswalk':
            msg = AckermannDriveStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = 'Cross_walk'
            msg.drive.steering_angle = 0.0
            msg.drive.speed = 0.0

            if now - self.start_time < 9.0:
                # 9초 안 지났으면 계속 정지 명령
                self.ack_pub_1.publish(msg)
                rospy.loginfo(f'정지 중 시간: {now - self.start_time}')
            else:
                self.state = 'done'   # 다시 None으로 안 돌려서 루프 방지
                print(f'정지 완료 시간: {now - self.start_time}')
                rospy.loginfo(f'정지 완료 시간: {now - self.start_time}')

                msg = Float64()
                msg.data = 3.
                rospy.loginfo('------mission 3------')
                self.current_mission = 3
                self.mission_pub.publish(msg)



        # 3) done 상태: 이 노드는 더 이상 간섭 안 함
        elif self.state == 'done':
            pass
        cv.waitKey(1)
if __name__ =='__main__':
    try:
        cw = Crosswalk()
        rate = rospy.Rate(25)  # 30Hz

        while not rospy.is_shutdown():
            if cw.current_mission == 2.0:
                cw.main()
                rate.sleep()

    except rospy.ROSInterruptException:
        pass