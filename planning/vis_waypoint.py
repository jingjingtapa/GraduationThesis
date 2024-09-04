import carla, pygame, math, random, sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from client.init import initializer

sim = initializer()

pygame.init()
display = pygame.display.set_mode((sim.DISPLAY_WIDTH, sim.DISPLAY_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption('Vehicle Close Waypoints')
clock = pygame.time.Clock()

def draw_vehicle(display, vehicle_transform, vehicle_width, vehicle_length):
    vehicle_heading = np.deg2rad(vehicle_transform.rotation.yaw)
    center_x, center_y= sim.DISPLAY_WIDTH // 2, sim.DISPLAY_HEIGHT // 2

    corners = []
    for dx, dy in [(-vehicle_length / 2, -vehicle_width / 2), (-vehicle_length / 2, vehicle_width / 2),
                   (vehicle_length / 2, vehicle_width / 2), (vehicle_length / 2, -vehicle_width / 2)]:
        rotated_x = dx * math.cos(vehicle_heading) - dy * math.sin(vehicle_heading)
        rotated_y = dx * math.sin(vehicle_heading) + dy * math.cos(vehicle_heading)
        corners.append((center_x + rotated_x * 30, center_y + rotated_y * 30))

    pygame.draw.polygon(display, sim.BLUE, corners, 0) 
    pygame.draw.polygon(display, sim.YELLOW, corners, 3)

def draw_close_waypoints(display, map, vehicle_transform, radius, vehicle_width, vehicle_length):
    display.fill((0, 0, 0))
    vehicle_location = vehicle_transform.location
    vehicle_heading = vehicle_transform.rotation.yaw

    close_waypoints = map.generate_waypoints(vehicle_length)
    for waypoint in close_waypoints:
        wp_location = waypoint.transform.location
        wp_heading = waypoint.transform.rotation.yaw
        distance = math.sqrt((wp_location.x - vehicle_location.x) ** 2 + (wp_location.y - vehicle_location.y) ** 2)

        heading_difference = abs((vehicle_heading - wp_heading + 180) % 360 - 180)
        
        if distance < radius and heading_difference < 90: # 반경 외 & 반대 차선 filtering
            lane_width = waypoint.lane_width / 2
            perpendicular_heading = np.deg2rad(wp_heading + 90)
            
            left_x = wp_location.x + lane_width * math.cos(perpendicular_heading)
            left_y = wp_location.y + lane_width * math.sin(perpendicular_heading)
            right_x = wp_location.x - lane_width * math.cos(perpendicular_heading)
            right_y = wp_location.y - lane_width * math.sin(perpendicular_heading)

            coordinate_scale = 30

            screen_wp_x = int((vehicle_location.x - wp_location.x) * coordinate_scale + sim.DISPLAY_WIDTH / 2)
            screen_wp_y = int((vehicle_location.y - wp_location.y) * coordinate_scale + sim.DISPLAY_HEIGHT / 2)
            screen_left_x = int((vehicle_location.x - left_x) * coordinate_scale + sim.DISPLAY_WIDTH / 2)
            screen_left_y = int((vehicle_location.y - left_y) * coordinate_scale + sim.DISPLAY_HEIGHT / 2)
            screen_right_x = int((vehicle_location.x - right_x) * coordinate_scale + sim.DISPLAY_WIDTH / 2)
            screen_right_y = int((vehicle_location.y - right_y) * coordinate_scale + sim.DISPLAY_HEIGHT / 2)

            pygame.draw.circle(display, sim.RED, (screen_wp_x, screen_wp_y), 5)
            pygame.draw.circle(display, sim.GREEN, (screen_left_x, screen_left_y), 5)
            pygame.draw.circle(display, sim.GREEN, (screen_right_x, screen_right_y), 5)

    draw_vehicle(display, vehicle_transform, vehicle_width, vehicle_length)
    pygame.display.flip()

def main():
    vehicle_bp = sim.blueprint_library.filter('vehicle.*')[0]
    
    spawn_point = random.choice(sim.world.get_map().get_spawn_points())
    vehicle = sim.world.spawn_actor(vehicle_bp, spawn_point)

    vehicle_bounding_box = vehicle.bounding_box 
    vehicle_width, vehicle_length = vehicle_bounding_box.extent.y * 2, vehicle_bounding_box.extent.x * 2

    vehicle.set_autopilot(True)
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            vehicle_transform = vehicle.get_transform()
            draw_close_waypoints(display, sim.world.get_map(), vehicle_transform, 10, vehicle_width, vehicle_length)
            clock.tick(30)

    finally:
        vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
