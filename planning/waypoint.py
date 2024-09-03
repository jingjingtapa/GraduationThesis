import carla
import pygame
import numpy as np
import math
import random

# PyGame 초기화
pygame.init()

# 색상 정의
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# PyGame 화면 초기화
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
clock = pygame.time.Clock()

# Carla 클라이언트 설정
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 웨이포인트 시각화 함수
def draw_waypoints(display, map, vehicle_location, scale=1.5, point_radius=3, close_point_radius=5):
    """
    전체 웨이포인트와 차량 주변의 웨이포인트를 시각화하는 함수
    """
    # 전체 웨이포인트 시각화
    waypoints = map.generate_waypoints(8.0)  # 전체 맵에 8미터 간격으로 waypoints 생성
    for waypoint in waypoints:
        wp_location = waypoint.transform.location
        x = int((wp_location.x - vehicle_location.x) * scale + DISPLAY_WIDTH / 2)
        y = int((vehicle_location.y - wp_location.y) * scale + DISPLAY_HEIGHT / 2)
        pygame.draw.circle(display, WHITE, (x, y), point_radius)

    # 차량 주변 웨이포인트 시각화
    close_waypoints = map.generate_waypoints(2.0)  # 차량 주변에 2미터 간격으로 waypoints 생성
    for waypoint in close_waypoints:
        wp_location = waypoint.transform.location
        distance = math.sqrt((wp_location.x - vehicle_location.x) ** 2 + (wp_location.y - vehicle_location.y) ** 2)
        if distance < 50:  # 50미터 이내
            x = int((wp_location.x - vehicle_location.x) * scale + DISPLAY_WIDTH / 2)
            y = int((vehicle_location.y - wp_location.y) * scale + DISPLAY_HEIGHT / 2)
            pygame.draw.circle(display, RED, (x, y), close_point_radius)

# 주 프로그램 함수
def main():
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    vehicle.set_autopilot(True)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            vehicle_location = vehicle.get_transform().location

            display.fill((0, 0, 0))  # 화면을 검은색으로 지움
            draw_waypoints(display, world.get_map(), vehicle_location)
            pygame.draw.circle(display, BLUE, (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 2), 7)  # 차량 위치
            pygame.display.flip()
            clock.tick(30)

    finally:
        vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
