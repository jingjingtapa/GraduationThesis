import carla
import pygame
import numpy as np
import math
import random

pygame.init()
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption('Vehicle Visualization')
clock = pygame.time.Clock()

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

def draw_vehicle(display, vehicle_transform, vehicle_width, vehicle_length):
    vehicle_location = vehicle_transform.location
    vehicle_heading = np.deg2rad(vehicle_transform.rotation.yaw)
    center_x = DISPLAY_WIDTH // 2
    center_y = DISPLAY_HEIGHT // 2

    # Calculate the four corners of the vehicle
    corners = []
    for dx, dy in [(-vehicle_length / 2, -vehicle_width / 2), (-vehicle_length / 2, vehicle_width / 2),
                   (vehicle_length / 2, vehicle_width / 2), (vehicle_length / 2, -vehicle_width / 2)]:
        rotated_x = dx * math.cos(vehicle_heading) - dy * math.sin(vehicle_heading)
        rotated_y = dx * math.sin(vehicle_heading) + dy * math.cos(vehicle_heading)
        corners.append((center_x + rotated_x * 30, center_y + rotated_y * 30))

    # Draw the vehicle as a polygon
    pygame.draw.polygon(display, BLUE, corners, 0)  # Fill color
    pygame.draw.polygon(display, YELLOW, corners, 3)  # Border color

def draw_close_waypoints(display, map, vehicle_transform, radius=10):
    display.fill((0, 0, 0))
    vehicle_location = vehicle_transform.location
    vehicle_heading = np.deg2rad(vehicle_transform.rotation.yaw)

    close_waypoints = map.generate_waypoints(2)
    for waypoint in close_waypoints:
        wp_location = waypoint.transform.location
        distance = math.sqrt((wp_location.x - vehicle_location.x) ** 2 + (wp_location.y - vehicle_location.y) ** 2)
        if distance < radius:
            lane_width = waypoint.lane_width / 2
            perpendicular_heading = np.deg2rad(waypoint.transform.rotation.yaw + 90)

            left_x = wp_location.x + lane_width * math.cos(perpendicular_heading)
            left_y = wp_location.y + lane_width * math.sin(perpendicular_heading)
            right_x = wp_location.x - lane_width * math.cos(perpendicular_heading)
            right_y = wp_location.y - lane_width * math.sin(perpendicular_heading)

            for x, y, color in [(left_x, left_y, GREEN), (right_x, right_y, GREEN)]:
                screen_x = int((vehicle_location.x - x) * 50 + DISPLAY_WIDTH / 2)
                screen_y = int((vehicle_location.y - y) * 50 + DISPLAY_HEIGHT / 2)
                pygame.draw.circle(display, color, (screen_x, screen_y), 5)

    draw_vehicle(display, vehicle_transform, 2.5, 5)
    pygame.display.flip()

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

            vehicle_transform = vehicle.get_transform()
            draw_close_waypoints(display, world.get_map(), vehicle_transform)
            clock.tick(30)

    finally:
        vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
