import glob
import os
import sys
from queue import Queue, Empty
import numpy as np
import pygame
import carla
import argparse

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

def sensor_callback(data, queue):
    """카메라 데이터를 큐에 저장하는 콜백 함수."""
    queue.put(data)

def tutorial(args):
    """앞뒤 카메라 데이터를 실시간으로 가져와 Pygame으로 표시."""
    pygame.init()
    display = pygame.display.set_mode((args.width * 2, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    vehicle, front_camera, rear_camera = None, None, None

    try:
        vehicle_bp = bp_lib.filter("vehicle.lincoln.mkz_2017")[0]
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]

        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))

        vehicle = world.spawn_actor(vehicle_bp, transform=world.get_map().get_spawn_points()[0])
        vehicle.set_autopilot(True)

        front_camera = world.spawn_actor(camera_bp, transform=carla.Transform(carla.Location(x=1.6, z=1.6)), attach_to=vehicle)
        rear_camera = world.spawn_actor(camera_bp, transform=carla.Transform(carla.Location(x=-1.6, z=1.6), carla.Rotation(yaw=180)), attach_to=vehicle)

        front_image_queue, rear_image_queue = Queue(), Queue()
        front_camera.listen(lambda data: sensor_callback(data, front_image_queue))
        rear_camera.listen(lambda data: sensor_callback(data, rear_image_queue))

        running = True
        while running:
            world.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            try:
                front_image_data = front_image_queue.get(True, 1.0)
                rear_image_data = rear_image_queue.get(True, 1.0)
            except Empty:
                continue

            front_im_array = np.frombuffer(front_image_data.raw_data, dtype=np.dtype("uint8")).reshape((front_image_data.height, front_image_data.width, 4))[:, :, :3][:, :, ::-1]
            rear_im_array = np.frombuffer(rear_image_data.raw_data, dtype=np.dtype("uint8")).reshape((rear_image_data.height, rear_image_data.width, 4))[:, :, :3][:, :, ::-1]

            front_surface = pygame.surfarray.make_surface(front_im_array.swapaxes(0, 1))
            rear_surface = pygame.surfarray.make_surface(rear_im_array.swapaxes(0, 1))

            display.blit(front_surface, (0, 0))
            display.blit(rear_surface, (args.width, 0))
            pygame.display.flip()

    finally:
        world.apply_settings(original_settings)
        if front_camera:
            front_camera.destroy()
        if rear_camera:
            rear_camera.destroy()
        if vehicle:
            vehicle.destroy()
        pygame.quit()

def main():
    argparser = argparse.ArgumentParser(description='CARLA Front and Rear Camera real-time visualization using Pygame')
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--res', default='680x420', help='window resolution (default: 1280x720)')
    args = argparser.parse_args()
    args.width, args.height = map(int, args.res.split('x'))

    try:
        tutorial(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()

