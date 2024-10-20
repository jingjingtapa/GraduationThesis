import carla
import pygame
import numpy as np

# Pygame 초기화
pygame.init()

# Pygame 디스플레이 설정
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
screen = pygame.display.set_mode((CAMERA_WIDTH, CAMERA_HEIGHT))

# CARLA와 연결
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Ego 차량 스폰 및 카메라 부착
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 카메라 부착 함수 정의
def attach_camera(vehicle, transform, fov=90):
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{CAMERA_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{CAMERA_HEIGHT}')
    camera_bp.set_attribute('fov', f'{fov}')
    camera_transform = carla.Transform(carla.Location(x=transform[0], y=transform[1], z=transform[2]),
                                       carla.Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5]))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

# 전방 카메라 부착 (차량의 앞쪽에 위치)
transform = [2.0, 0.0, 1.5, 0.0, 0.0, 0.0]  # x, y, z, pitch, yaw, roll
front_camera = attach_camera(vehicle, transform)

# 카메라에서 받은 데이터를 저장할 변수
image_front = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

# 콜백 함수: 카메라로부터 받은 이미지 처리
def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # RGB만 추출
    array = array[:, :, ::-1]  # BGR -> RGB 변환
    global image_front
    image_front = array

# 콜백 함수 등록
front_camera.listen(lambda image: process_image(image))

# Pygame을 통한 실시간 시각화 루프
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 카메라 이미지가 있으면 Pygame에 렌더링
    if image_front is not None:
        surface = pygame.surfarray.make_surface(image_front.swapaxes(0, 1))
        screen.blit(surface, (0, 0))

    pygame.display.flip()

# 종료 시 actor 정리
front_camera.stop()
vehicle.destroy()
pygame.quit()