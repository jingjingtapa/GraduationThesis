import cv2
import numpy as np
import matplotlib.pyplot as plt

# 전면 및 후면 카메라 이미지 경로
front_image_path = "/home/gunny/다운로드/GraduationThesis/Carla_BEV/image/front_camera_frame_0.png"
rear_image_path = "/home/gunny/다운로드/GraduationThesis/Carla_BEV/image/rear_camera_frame_0.png"

# 이미지 불러오기
front_image = cv2.imread(front_image_path)
rear_image = cv2.imread(rear_image_path)

# 전면 및 후면 카메라의 소스 좌표 정의 (이미지의 도로 부분에 맞춰 수동으로 설정)
src_points_front = np.float32([[150, 300], [550, 300], [50, 400], [650, 400]])
src_points_rear = np.float32([[150, 300], [550, 300], [50, 400], [650, 400]])  # 후면 이미지도 동일한 좌표를 적용

# 변환할 목적지 좌표 정의 (Bird's Eye View 상에서 대응되는 좌표)
dst_points = np.float32([[200, 0], [520, 0], [200, 720], [520, 720]])

# 전면 이미지의 호모그래피 행렬 계산 및 BEV 변환
H_front, _ = cv2.findHomography(src_points_front, dst_points)
bev_front_image = cv2.warpPerspective(front_image, H_front, (front_image.shape[1], front_image.shape[0]))

# 후면 이미지의 호모그래피 행렬 계산 및 BEV 변환
H_rear, _ = cv2.findHomography(src_points_rear, dst_points)
bev_rear_image = cv2.warpPerspective(rear_image, H_rear, (rear_image.shape[1], rear_image.shape[0]))

# 후면 이미지를 180도 회전 (좌우 및 상하 뒤집기)
bev_rear_image = cv2.rotate(bev_rear_image, cv2.ROTATE_180)

# 좌우 이동을 위한 변환 행렬 생성 (x_shift 값에 따라 이동량 결정)
x_shift = 27  # 좌우 이동량 (양수: 오른쪽으로 이동, 음수: 왼쪽으로 이동)
translation_matrix = np.float32([[1, 0, x_shift], [0, 1, 0]])

# 후면 이미지를 좌우로 이동
bev_rear_image_shifted = cv2.warpAffine(bev_rear_image, translation_matrix, (bev_rear_image.shape[1], bev_rear_image.shape[0]))

# 전면과 후면 이미지의 크기가 일치하는지 확인
front_height, front_width = bev_front_image.shape[:2]
rear_height, rear_width = bev_rear_image_shifted.shape[:2]

# 전면 이미지와 후면 이미지의 너비가 다르면 너비를 맞추기
if front_width != rear_width:
    bev_rear_image_shifted = cv2.resize(bev_rear_image_shifted, (front_width, rear_height))

# 이어 붙일 때 offset을 사용하여 미세 조정
combined_image = np.vstack((bev_front_image, bev_rear_image_shifted))

# 결합된 이미지 시각화
plt.figure(figsize=(8, 12))  # 세로로 길게 설정
plt.title('Combined BEV: Front + Rear (Shifted Rear)')
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # 축 없애기
plt.show()






