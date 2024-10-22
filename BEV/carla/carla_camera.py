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
screen_width = int((wy_max-wy_min)/wy_interval)
screen_height = int((wx_max-wx_min)/wx_interval)*2
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
    
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(4)
    # R[0, 0] = cj * ck
    # R[0, 1] = sj * sc - cs
    # R[0, 2] = sj * cc + ss
    # R[1, 0] = cj * sk
    # R[1, 1] = sj * ss + cc
    # R[1, 2] = sj * cs - sc
    # R[2, 0] = -sj
    # R[2, 1] = cj * si
    # R[2, 2] = cj * ci
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

# 차량의 물리 파라미터 설정을 가져오기
vehicle_physics_control = vehicle.get_physics_control()

# 서스펜션 강도와 댐퍼 값을 조정하여 피칭을 줄임
for wheel in vehicle_physics_control.wheels:
    wheel.suspension_stiffness = 100.0  # 서스펜션을 더 강하게 설정
    wheel.wheel_damping_rate = 0.2  # 댐퍼 값을 낮게 설정하여 흔들림 줄이기

# 차량의 질량 중심을 낮게 설정하여 피칭을 줄임
vehicle_physics_control.center_of_mass = carla.Vector3D(0.0, 0.0, -0.5)  # Z축을 낮게 설정하여 무게중심을 아래로 내림

# 새로운 물리 파라미터를 차량에 적용
vehicle.apply_physics_control(vehicle_physics_control)

vehicle.set_autopilot(True)

# 카메라 부착 함수 정의
def attach_camera(vehicle, transform, fov=FOV):
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{CAMERA_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{CAMERA_HEIGHT}')
    camera_bp.set_attribute('fov', f'{fov}')
    camera_transform = carla.Transform(carla.Location(x=transform[0], y=transform[1], z=transform[2]),
                                       carla.Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5]))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

# 카메라 부착
transform = [2.0, 0.0, 1.5, 0.0, 0.0, 0.0]
front_camera = attach_camera(vehicle, transform)

K = intrinsic(CAMERA_WIDTH, CAMERA_HEIGHT, FOV)
R = extrinsic(transform)

map_x, map_y, bev_height, bev_width = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval,
                                                                           wy_min, wy_max, wy_interval,
                                                                           R, K)

# 각 카메라에서 받은 데이터를 화면에 그리기 위한 배열
image_front = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)

# 콜백 함수: 카메라로부터 받은 이미지 처리
def process_image(image, image_type):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # BGR만 추출
    array = cv2.remap(array, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    array = array[:, :, ::-1]  # BGR -> RGB 변환
    if image_type == 'front':
        global image_front
        image_front = array

# 콜백 함수 등록
front_camera.listen(lambda image: process_image(image, 'front'))

# Pygame을 통한 실시간 시각화 루프
running = True
cnt = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 화면 배치 설정 (4개 카메라에서 받은 투영 변환 이미지를 하나로 결합)
    screen.fill((0, 0, 0))  # 화면 초기화
    if image_front is not None and image_front.size > 0:
        # cv2는 기본적으로 (height, width, channels) 형식이므로 swapaxes를 사용해 Pygame에서 렌더링할 수 있도록 변경
        surface = pygame.surfarray.make_surface(image_front.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
    else:
        print("Front view is empty or None")

    pygame.display.flip()

# 종료 시 actor 정리
front_camera.stop()
vehicle.destroy()
pygame.quit()
