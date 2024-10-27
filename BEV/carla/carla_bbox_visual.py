import carla
import pygame
import numpy as np

# CARLA 클라이언트 연결
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Blueprint 라이브러리에서 Tesla 차량 가져오기
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

# Ego 차량 스폰
spawn_point = world.get_map().get_spawn_points()[0]
ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 카메라 설정
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

# Pygame 설정
pygame.init()
display = pygame.display.set_mode((800, 600))

# 이미지 처리 및 Pygame 디스플레이에 그리기
def process_img(image):
    # 디버그 메시지 출력
    print("Processing image...")

    # 이미지를 넘파이 배열로 변환
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]  # RGB만 사용
    
    # 이미지를 Pygame 디스플레이에 업데이트
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    pygame.display.flip()

# 이미지 수신 이벤트에 process_img 함수 연결
camera.listen(lambda image: process_img(image))

# 시뮬레이션 루프
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
finally:
    # 종료 시 모든 액터 삭제
    camera.stop()
    ego_vehicle.destroy()
    pygame.quit()




