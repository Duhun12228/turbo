# turbo

1️⃣ 클래스 초기화 (__init__)

ROS 노드 초기화: rospy.init_node('lane_follow')

CvBridge: ROS 이미지 메시지를 OpenCV 이미지로 변환

Subscriber / Publisher

/usb_cam/image_rect_color/compressed → 카메라 이미지 구독

/commands/motor/speed → 모터 속도 발행

/commands/servo/position → 조향 서보 발행

/debugging_image1, /debugging_image2 → 디버깅용 이미지 발행

차선 검출 HSV 범위: 흰색만 필터링

투시 변환 (버드아이 뷰)

src_points → 원본 이미지 좌표

dst_points → 직사각형 변환 좌표

matrix, inv_matrix → 변환행렬 및 역변환

이미지, PID, 속도, 서보/모터 파라미터 초기화

2️⃣ 이미지 콜백 (image_cb)

매 프레임마다 실행되는 핵심 루프

이미지 압축 해제

bgr = self.cv_bridge.compressed_imgmsg_to_cv2(image_msg, 'bgr8')


화이트 필터링

HSV 변환 → 범위 내 픽셀 추출

미디언 블러, 모폴로지 닫기 적용

(추가로 Gaussian Blur도 적용 가능)

버드아이 변환

흰색 마스크와 원본 이미지를 투시 변환

슬라이딩 윈도우 + 폴리노미얼 회귀 (sliding_window_polyfit)

히스토그램으로 초기 차선 위치 탐지

각 윈도우별 픽셀 선택

2차 곡선 회귀 (polyfit) → left_fit, right_fit

차선 중심 계산

양쪽 차선 좌표 평균 → lane_center

차량 중심과의 오차 계산

PID 제어로 스티어링 계산 (pid_control)

속도 조절 (adjust_speed)

곡선 구간에서 감속

서보 / 모터 명령 발행

servo_cmd = self.steer_to_servo_gain * steer_angle + offset
motor_cmd = speed * self.speed_to_erpm_gain + offset


디버깅 이미지 생성 및 발행

create_debug_image → 슬라이딩 윈도우 영역, 차선 픽셀 표시

create_lane_overlay → 원본 위 반투명 차선 영역 표시

3️⃣ 슬라이딩 윈도우 + 회귀 (sliding_window_polyfit)

이미지 아래쪽 히스토그램으로 차선 시작점 탐색

윈도우를 위로 이동하며 차선 픽셀 선택

픽셀 수 기준(minpix)으로 윈도우 이동

2차 곡선 회귀:

left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

4️⃣ 차선 중심 계산 (calculate_lane_center)

양쪽 차선의 y 위치에서 x 좌표 계산

양쪽 평균 → lane_center

한쪽 차선만 있을 경우: fallback or 예측 기능 사용 가능

5️⃣ PID 제어 (pid_control)

오차 기반 PID 계산

저번 값과 혼합하여 스티어링 부드럽게 조정

반환값 → 스티어링 각도

6️⃣ 속도 조절 (adjust_speed)

곡선 시 감속

최소/최대 속도 제한

7️⃣ 디버깅 이미지

슬라이딩 윈도우 / 픽셀 표시 (create_debug_image)

왼쪽/오른쪽 픽셀 → 색깔 표시 (빨강/파랑)

최근 픽셀 좌표 기반

차선 영역 오버레이 (create_lane_overlay)

왼쪽/오른쪽 폴리노미얼 계산

좌우 차선 영역 채우기

버드아이 → 원본 변환

반투명 합성

8️⃣ 예측 기능 (구상)

문제 상황: 한쪽 차선만 보일 때

해결 방법:

최근 3초(예: 30프레임) 차선 좌표 deque에 저장

한쪽 차선만 있을 때, 기존 좌표 + 2차 회귀 → 예측 차선 생성

디버깅 이미지에서 예측 차선 색을 실제 차선과 다르게 표시

차량 중심 계산 시 예측 차선 포함 가능

9️⃣ 핵심 구조 요약

입력: 카메라 이미지

처리:

흰색 필터 → 블러 → 버드아이 변환

슬라이딩 윈도우 → 픽셀 선택 → 2차 회귀

최근 차선 데이터 저장 (deque) → 예측 차선 계산

차량 중심, 오차 계산 → PID 스티어링

속도 조절

출력:

서보, 모터 명령

디버깅 이미지 2개

슬라이딩 윈도우 / 픽셀 표시

차선 영역 오버레이 (실제 / 예측 구분)
