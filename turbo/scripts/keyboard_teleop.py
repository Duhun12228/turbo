                    #!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
import curses

class KeyboardVESCController:
    def __init__(self):
        rospy.init_node('keyboard_vesc_controller', anonymous=True)

        self.motor_pub = rospy.Publisher('/commands/motor/speed', Float64, queue_size=1)
        self.servo_pub = rospy.Publisher('/commands/servo/position', Float64, queue_size=1)

        # 변환 계수
        self.speed_to_erpm_gain = rospy.get_param("~speed_to_erpm_gain", 4600.0)
        self.speed_to_erpm_offset = rospy.get_param("~speed_to_erpm_offset", 0.0)
        self.steering_to_servo_gain = rospy.get_param("~steering_angle_to_servo_gain", -1.2135)
        self.steering_to_servo_offset = rospy.get_param("~steering_angle_to_servo_offset", 0.5304)

        # 단일 속도/조향 값
        self.speed_step = 0.5      # m/s (키 누르면 이 속도로 이동)
        self.steering_step = 0.5   # rad (키 누르면 이 각도로 이동)

        rospy.loginfo("Keyboard VESC Controller Started (Press arrow keys)")

    def run(self, stdscr):
        curses.cbreak()
        stdscr.keypad(True)
        stdscr.nodelay(True)  # 입력 대기 없이 진행

        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            key = stdscr.getch()
            speed = 0.0
            steering = 0.0

            # 키 체크
            if key == curses.KEY_UP:
                speed = self.speed_step
            elif key == curses.KEY_DOWN:
                speed = -self.speed_step
            elif key == curses.KEY_LEFT:
                steering = self.steering_step
            elif key == curses.KEY_RIGHT:
                steering = -self.steering_step
            elif key == 3:  # Ctrl+c
                break

            # 변환
            erpm = self.speed_to_erpm_gain * speed + self.speed_to_erpm_offset
            servo_pos = self.steering_to_servo_gain * steering + self.steering_to_servo_offset

            # 퍼블리시
            self.motor_pub.publish(Float64(erpm))
            self.servo_pub.publish(Float64(servo_pos))

            stdscr.addstr(0, 0, f"Speed: {speed:.2f} m/s -> ERPM: {erpm:.1f}, Steering: {steering:.2f} rad -> Servo: {servo_pos:.3f}  ")
            stdscr.refresh()
            rate.sleep()


if __name__ == "__main__":
    try:
        controller = KeyboardVESCController()
        curses.wrapper(controller.run)
    except rospy.ROSInterruptException:
        pass
