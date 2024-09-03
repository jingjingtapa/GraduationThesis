import carla
import pygame
import numpy as np
import math
import random

# Pygame 초기화
pygame.init()

# 색상 정의
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption('Vehicle Close Waypoints')
clock = pygame.time.Clock()

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

def draw_vehicle(display, vehicle_transform, vehicle_width, vehicle_length):
    vehicle_location = vehicle_transform.location
    vehicle_heading = np.deg2rad(vehicle_transform.rotation.yaw)
    center_x = DISPLAY_WIDTH // 2
    center_y = DISPLAY_HEIGHT // 2

    # 차량의 네 모서리 계산
    corners = []
    for dx, dy in [(-vehicle_length / 2, -vehicle_width / 2), (-vehicle_length / 2, vehicle_width / 2),
                   (vehicle_length / 2, vehicle_width / 2), (vehicle_length / 2, -vehicle_width / 2)]:
        rotated_x = dx * math.cos(vehicle_heading) - dy * math.sin(vehicle_heading)
        rotated_y = dx * math.sin(vehicle_heading) + dy * math.cos(vehicle_heading)
        corners.append((center_x + rotated_x * 30, center_y + rotated_y * 30))

    # 차량을 다각형으로 그리기
    pygame.draw.polygon(display, BLUE, corners, 0)  # 채움색
    pygame.draw.polygon(display, YELLOW, corners, 3)  # 테두리 색상

def draw_close_waypoints(display, map, vehicle_transform, radius, vehicle_width, vehicle_length):
    display.fill((0, 0, 0))
    vehicle_location = vehicle_transform.location
    vehicle_heading = vehicle_transform.rotation.yaw

    close_waypoints = map.generate_waypoints(vehicle_length)
    for waypoint in close_waypoints:
        wp_location = waypoint.transform.location
        wp_heading = waypoint.transform.rotation.yaw
        distance = math.sqrt((wp_location.x - vehicle_location.x) ** 2 + (wp_location.y - vehicle_location.y) ** 2)

        # 방향성 조정을 위한 헤딩 차이 계산
        heading_difference = abs((vehicle_heading - wp_heading + 180) % 360 - 180)
        
        if distance < radius and heading_difference < 90:  # 같은 방향으로 간주되는 90도 이내
            lane_width = waypoint.lane_width / 2
            perpendicular_heading = np.deg2rad(wp_heading + 90)
            
            left_x = wp_location.x + lane_width * math.cos(perpendicular_heading)
            left_y = wp_location.y + lane_width * math.sin(perpendicular_heading)
            right_x = wp_location.x - lane_width * math.cos(perpendicular_heading)
            right_y = wp_location.y - lane_width * math.sin(perpendicular_heading)

            coordinate_scale = 30

            # 스크린 좌표로 변환
            screen_wp_x = int((vehicle_location.x - wp_location.x) * coordinate_scale + DISPLAY_WIDTH / 2)
            screen_wp_y = int((vehicle_location.y - wp_location.y) * coordinate_scale + DISPLAY_HEIGHT / 2)
            screen_left_x = int((vehicle_location.x - left_x) * coordinate_scale + DISPLAY_WIDTH / 2)
            screen_left_y = int((vehicle_location.y - left_y) * coordinate_scale + DISPLAY_HEIGHT / 2)
            screen_right_x = int((vehicle_location.x - right_x) * coordinate_scale + DISPLAY_WIDTH / 2)
            screen_right_y = int((vehicle_location.y - right_y) * coordinate_scale + DISPLAY_HEIGHT / 2)

            pygame.draw.circle(display, RED, (screen_wp_x, screen_wp_y), 5)
            pygame.draw.circle(display, GREEN, (screen_left_x, screen_left_y), 5)
            pygame.draw.circle(display, GREEN, (screen_right_x, screen_right_y), 5)

    draw_vehicle(display, vehicle_transform, vehicle_width, vehicle_length)
    pygame.display.flip()

def main():
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    vehicle_bounding_box = vehicle.bounding_box  # 차량의 3D 경계 상자를 가져옵니다.
    vehicle_width = vehicle_bounding_box.extent.y * 2  # 경계 상자의 'y' 축 길이는 폭의 반을 나타냅니다.
    vehicle_length = vehicle_bounding_box.extent.x * 2  # 경계 상자의 'x' 축 길이는 길이의 반을 나타냅니다.

    vehicle.set_autopilot(True)
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            vehicle_transform = vehicle.get_transform()
            draw_close_waypoints(display, world.get_map(), vehicle_transform, 10, vehicle_width, vehicle_length)
            clock.tick(30)

    finally:
        vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
