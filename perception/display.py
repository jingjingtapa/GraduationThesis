import carla
import numpy as np
import pygame
import time
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Pygame 및 OpenGL 초기화
pygame.init()

# 클라이언트 연결 (localhost, 기본 포트 2000)
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# 월드를 불러옴
world = client.get_world()

# 스폰 매니저 가져오기
blueprint_library = world.get_blueprint_library()

# 스폰할 차량을 설정
vehicle_bp = blueprint_library.filter('model3')[0]  # Tesla Model3 선택
spawn_point = world.get_map().get_spawn_points()[1]  # 첫 번째 스폰 지점

# 차량 스폰
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 카메라 센서 설정
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '110')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # 차량 앞쪽에 카메라 위치 설정

# 카메라 부착
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Pygame 디스플레이 설정 (OpenGL 모드)
display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.OPENGL | pygame.DOUBLEBUF)

# OpenGL 설정
glEnable(GL_TEXTURE_2D)
texture = glGenTextures(1)

# OpenGL 초기 설정
def init_opengl():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, 800, 600, 0)  # 2D 화면 좌표계 설정
    glMatrixMode(GL_MODELVIEW)

init_opengl()

# 카메라 데이터 콜백 함수
def process_image(image):
    image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
    image_data = image_data.reshape((image.height, image.width, 4))  # BGRA 포맷
    image_data = image_data[:, :, :3]  # BGR로 변환

    # OpenGL 텍스처로 이미지를 로드
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

    # 화면에 텍스처 그리기
    glClear(GL_COLOR_BUFFER_BIT)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0)
    glVertex2f(0, 0)
    glTexCoord2f(1, 0)
    glVertex2f(800, 0)
    glTexCoord2f(1, 1)
    glVertex2f(800, 600)
    glTexCoord2f(0, 1)
    glVertex2f(0, 600)
    glEnd()

    pygame.display.flip()

# 카메라에서 얻은 데이터를 실시간으로 처리
camera.listen(lambda image: process_image(image))

# 몇 초간 시뮬레이션이 유지되도록 대기
try:
    while True:
        # Pygame 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
        time.sleep(0.01)
finally:
    # 종료 시 모든 액터 정리
    camera.stop()
    vehicle.destroy()
    pygame.quit()
