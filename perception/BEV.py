import carla
import pygame
import math
import random
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from client.init import initializer

sim = initializer()
pygame.init()
display = pygame.display.set_mode((sim.DISPLAY_WIDTH, sim.DISPLAY_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF )
pygame.display.set_caption('Bird Eye View')
clock = pygame.time.Clock()

vehicle_bp = sim.blueprint_library.filter('vehicle.*')[0]

def camera_callback(image):
    image = sim.process_image(image)
    surface = pygame.surfarray.make_surface(image.swapaxes(0,1))
    display.blit(surface, (0, 0))
    pygame.display.flip()

def main():
    spawn_point = random.choice(sim.world.get_map().get_spawn_points())
    vehicle = sim.world.spawn_actor(vehicle_bp, spawn_point)
    v_bbox = vehicle.bounding_box 
    v_width, v_length = v_bbox.extent.y * 2, v_bbox.extent.x * 2

    camera = sim.attach_rear_camera(sim.world, vehicle)
    camera.listen(camera_callback)
    try:
        while True:
            for event in pygame.event.get():  # 종료 이벤트 처리
                if event.type == pygame.QUIT:
                    return            
                
            clock.tick(30)  # 화면 갱신 속도 조절
    finally:
        camera.stop()
        vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
