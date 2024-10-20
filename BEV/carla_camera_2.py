import carla
import pygame
import numpy as np
import cv2
from hist.BEVConverter import BEVConverter

wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = 7, 40, 0.05, -10, 10, 0.05

bev = BEVConverter(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval)

# Pygame 초기화
pygame.init()

# Pygame 디스플레이 설정
screen_width = int((wy_max - wy_min) / wy_interval)
screen_height = int((wx_max - wx_min) / wx_interval) * 2
screen = pygame.display.set_mode((screen_width, screen_height))

# 카메라 설정
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FOV = 90

def intrinsic(camera_width, camera_height, fov):
    f_x = camera_width / (2 * np.tan(np.radians(fov / 2)))
    f_y = f_x
    c_x, c_y = camera_width / 2, camera_height / 2

    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    return K

def extrinsic(t):
    x, y, z = t[0], t[1], t[2]
    roll, pitch, yaw = t[3], t[4], t[5]
    
    R = np.identity(4)
    R[0, 3], R[1, 3], R[2, 3] = -x, -y, -z

    r = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])
    
    RT = r @ R
    return RT

# Carla 시뮬레이터와 연결
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Ego 차량 스폰 및 카메라 부착
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 차량의 물리 파라미터 가져오기 (차량 길이 계산)
vehicle_physics_control = vehicle.get_physics_control()
vehicle_dimensions = vehicle_physics_control.wheels[0].position - vehicle_physics_control.wheels[2].position
vehicle_length = abs(vehicle_dimensions.x)

# 차량의 실제 길이를 wx_interval로 나눈 픽셀 간격 계산
pixel_gap = int(vehicle_length / wx_interval)

# 카메라 부착 (전방 카메라)
front_transform = [2.0, 0.0, 1.5, 0.0, 0.0, 0.0]
front_camera = attach_camera(vehicle, front_transform)

# 카메라 부착 (후방 카메라)
rear_transform = [-2.0, 0.0, 1.5, 0.0, 180.0, 0.0]
rear_camera = attach_camera(vehicle, rear_transform)

K = intrinsic(CAMERA_WIDTH, CAMERA_HEIGHT, FOV)
R_front = extrinsic(front_transform)
R_rear = extrinsic(rear_transform)

map_x_front, map_y_front, bev_height, bev_width = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval,
                                                                                      wy_min, wy_max, wy_interval,
                                                                                      R_front, K)

map_x_rear, map_y_rear, _, _ = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval,
                                                                    wy_min, wy_max, wy_interval,
                                                                    R_rear, K)

# 각 카메라에서 받은 데이터를 화면에 그리기 위한 배열
image_front = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
image_rear = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)

# 콜백 함수: 카메라로부터 받은 이미지 처리
def process_image(image, image_type):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # BGR만 추출
    if image_type == 'front':
        bev_image = cv2.remap(array, map_x_front, map_y_front, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        bev_image = bev_image[:, :, ::-1]  # BGR -> RGB 변환
        global image_front
        image_front = bev_image
    
    elif image_type == 'rear':
        bev_image = cv2.remap(array, map_x_rear, map_y_rear, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        bev_image = bev_image[:, :, ::-1]  # BGR -> RGB 변환

        # 후방 이미지를 전방 이미지와 같은 크기로 맞춤 (리사이즈)
        if bev_image.shape != image_front.shape:
            bev_image = cv2.resize(bev_image, (image_front.shape[1], image_front.shape[0]))

        global image_rear
        image_rear = bev_image

# 콜백 함수 등록
front_camera.listen(lambda image: process_image(image, 'front'))
rear_camera.listen(lambda image: process_image(image, 'rear'))

# Pygame을 통한 실시간 시각화 루프
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 화면 초기화
    screen.fill((0, 0, 0))

    # 전방과 후방 카메라 이미지 결합 (차량 길이만큼 띄워서)
    if image_front is not None and image_front.size > 0 and image_rear is not None and image_rear.size > 0:
        # 후방 이미지를 상하좌우 모두 뒤집기
        image_rear_flipped = cv2.flip(image_rear, -1)  # 상하좌우 반전

        # 차량 길이에 해당하는 픽셀만큼의 간격을 추가하여 결합
        padding = np.zeros((pixel_gap, image_front.shape[1], 3), dtype=np.uint8)
        combined_image = np.vstack((image_front, padding, image_rear_flipped))

        # Pygame에서 렌더링할 수 있도록 변환
        surface = pygame.surfarray.make_surface(combined_image.swapaxes(0, 1))
        screen.blit(surface, (0, 0))

    pygame.display.flip()

# 종료 시 actor 정리
front_camera.stop()
rear_camera.stop()
vehicle.destroy()
pygame.quit()