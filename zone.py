import rospy
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float64
from ackermann_msgs.msg import AckermannDriveStamped
import math
import time

class Zone():
    def __init__(self):
        rospy.init_node('zone')
        self.cv_bridge = CvBridge()
        self.roi_img_sub = rospy.Subscriber('/roi_img', Image, self.roi_img_cb, queue_size=1)
        self.mission_pub = rospy.Publisher('/mission_num', Float64, queue_size=10)
        self.mission_sub = rospy.Subscriber('/mission_num', Float64, self.mission_cb, queue_size=1)
        self.ack_pub_1 = rospy.Publisher('/high_level/ackermann_cmd_mux/input/nav_1', AckermannDriveStamped, queue_size=10)
        self.red_lower1 = np.array(rospy.get_param('~red_lower1', [0, 50, 80]), dtype=np.uint8)
        self.red_upper1 = np.array(rospy.get_param('~red_upper1', [8, 255, 255]), dtype=np.uint8)

        self.red_lower2 = np.array(rospy.get_param('~red_lower2', [170, 50, 80]), dtype=np.uint8)
        self.red_upper2 = np.array(rospy.get_param('~red_upper2', [180, 255, 255]), dtype=np.uint8)

        self.blue_lower = np.array(rospy.get_param('~blue_lower',[95, 40, 80]), dtype=np.uint8)
        self.blue_upper = np.array(rospy.get_param('~blue_upper', [125, 255, 255]), dtype=np.uint8)
        
        self.current_mission = 1.
        self.roi_img = None
        self.zone = None
        self.state = None
    def roi_img_cb(self, msg):
        self.roi_img = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def mission_cb(self,msg):
        self.current_mission= msg.data
    
    def red_color_filter_hsv(self,img):

        bilateral = cv.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)
        # 3. HSV 변환
        hsv = cv.cvtColor(bilateral, cv.COLOR_BGR2HSV)

        # 5. inRange 마스크 생성
        mask1 = cv.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv.inRange(hsv, self.red_lower2, self.red_upper2)
        mask = cv.bitwise_or(mask1, mask2)

        # 6. 마스크 노이즈 제거 (반사 억제)
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel)

        # 7. 빨간 부분만 추출
        red_only = cv.bitwise_and(img, img, mask=mask_clean) 

        return red_only
    #반사 등 제거 후 명확하게
    def binary_filter(self,img):
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        _,binary =  cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
        return binary
    def blue_color_filter_hsv(self,img):
        # 2. Bilateral 필터 (반사 억제)
        bilateral = cv.bilateralFilter(img, d=7, sigmaColor=75, sigmaSpace=75)

        # 3. HSV 변환
        hsv = cv.cvtColor(bilateral, cv.COLOR_BGR2HSV)
        
        # 5. inRange 마스크 생성
        mask = cv.inRange(hsv, self.blue_lower, self.blue_upper)

        # 6. 마스크 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask_clean = cv.morphologyEx(mask_clean, cv.MORPH_CLOSE, kernel)
        
        # 7. 블루 영역만 추출
        blue_only = cv.bitwise_and(img, img, mask=mask_clean)

        return blue_only

    def detect_zone(self,img): #warp된 bgr 이미지
        if img is None:
            return None
        
        red_img = self.red_color_filter_hsv(img)
        blue_img = self.blue_color_filter_hsv(img)
        
        bin_red_img = self.binary_filter(red_img)
        bin_blue_img = self.binary_filter(blue_img)

        bin_red_img = bin_red_img[:,180:520]
        bin_blue_img = bin_blue_img[:,180:520]
        cv.imshow('crop_red_img',bin_red_img)
        ratio_red = cv.countNonZero(bin_red_img) / bin_red_img.size
        ratio_blue = cv.countNonZero(bin_blue_img) / bin_blue_img.size
        rospy.loginfo(f'ratio_red: {ratio_red}%, ratio_blue: {ratio_blue}%')
        if ratio_red > 0.5:
            return 'red'
        elif ratio_blue > 0.5:
            return 'blue'
        else:
            return None

    def publish_ack(self, speed, steering = 0.0):
        ack = AckermannDriveStamped()
        ack.header.stamp = rospy.Time.now()
        ack.drive.speed = speed
        ack.drive.steering_angle = steering
        self.ack_pub_1.publish(ack)

    def main(self):
        if self.state == None:   
            if self.detect_zone(self.roi_img) == None:
                    pass
            elif self.detect_zone(self.roi_img) == 'red':
                    self.publish_ack(speed = 0.22)
                    self.state = 'red'
                    rospy.loginfo('Red zone is detected!')
        
            
        elif self.state == 'red' and self.detect_zone(self.roi_img) == 'red':
            self.publish_ack(speed = 0.22)
            rospy.loginfo('In Redzone!')

        elif self.state == 'red' and self.detect_zone(self.roi_img) == None:
            self.publish_ack(speed = 0.22)
            rospy.loginfo('Something Error... Still maybe in redzone!')        

        elif self.state == 'red' and self.detect_zone(self.roi_img) == 'blue':
            self.publish_ack(speed = 0.5)
            rospy.loginfo('Blue zone is detected!')
            self.state = 'blue'
        
        elif self.state =='blue' and self.detect_zone(self.roi_img) == 'blue':
            self.publish_ack(speed = 0.5)
            rospy.loginfo('In Bluezone!')

        elif self.state == 'blue' and self.detect_zone(self.roi_img) == None:
            self.state = 'Done'
            rospy.loginfo('Zone detection is over!')

        elif self.state == 'Done':
            msg = Float64()
            msg.data = 2.
            self.current_mission = 2.
            self.mission_pub.publish(msg)
            rospy.loginfo('------mission 2------')

if __name__ =='__main__':
    try:
        zone = Zone()
        rate = rospy.Rate(25)  # 25Hz

        while not rospy.is_shutdown():
            if zone.current_mission == 1.0:
                zone.main()
                rate.sleep()

    except rospy.ROSInterruptException:
        pass