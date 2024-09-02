import carla
import random
import time
import pygame
import numpy as np

# Pygame을 사용하여 이미지를 화면에 표시
def display_camera(image, display):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # RGB로 변환
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    pygame.display.flip()

# Pygame 초기화
pygame.init()
display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)

try:
    # CARLA 서버에 연결
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # World와 Blueprint Library 가져오기
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # 랜덤 차량 Blueprint 선택
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

    # 스폰 위치 설정 (기본 spawn points 가져오기)
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)

    # 차량 스폰
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle is not None:
        print(f'Vehicle spawned: {vehicle.type_id}')

        # 차량에 카메라 센서 추가
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '110')

        # 카메라의 위치 설정 (차량 앞쪽에 부착)
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # 차량의 앞쪽 위에 카메라 설치
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # 카메라의 이미지 처리 콜백
        camera.listen(lambda image: display_camera(image, display))

        # Autopilot 모드 활성화
        vehicle.set_autopilot(True)

        # 주행 화면을 20초 동안 표시
        for _ in range(200):
            world.tick()  # 서버의 시간 동기화
            pygame.event.pump()

        # 종료
        camera.stop()
        camera.destroy()
        vehicle.set_autopilot(False)
        vehicle.destroy()
        print('Vehicle and camera destroyed')

finally:
    pygame.quit()
