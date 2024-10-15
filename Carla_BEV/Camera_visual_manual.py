#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Real-time manual control and visualization of front and rear cameras using Pygame
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
from queue import Queue, Empty
import numpy as np
import pygame
from pygame.locals import K_w, K_s, K_a, K_d, K_SPACE, K_q, K_r, K_ESCAPE


def sensor_callback(data, queue):
    """
    Callback to store the sensor data in a queue.
    """
    queue.put(data)


class ManualControl:
    """
    ManualControl class to handle vehicle control using keyboard inputs
    """
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.control = carla.VehicleControl()
        self.steer_cache = 0.0
        self.vehicle.set_autopilot(False)

    def parse_events(self):
        keys = pygame.key.get_pressed()

        if keys[K_w]:
            self.control.throttle = min(self.control.throttle + 0.1, 1.0)
        else:
            self.control.throttle = 0.0

        if keys[K_s]:
            self.control.brake = min(self.control.brake + 0.1, 1.0)
        else:
            self.control.brake = 0.0

        steer_increment = 5e-4 * 60
        if keys[K_a]:
            if self.steer_cache > 0:
                self.steer_cache = 0
            else:
                self.steer_cache -= steer_increment
        elif keys[K_d]:
            if self.steer_cache < 0:
                self.steer_cache = 0
            else:
                self.steer_cache += steer_increment
        else:
            self.steer_cache = 0.0

        self.steer_cache = min(0.7, max(-0.7, self.steer_cache))
        self.control.steer = self.steer_cache

        self.control.hand_brake = keys[K_SPACE]

        # R key to toggle reverse gear
        if keys[K_r]:
            self.control.gear = -1  # Set to reverse gear
        else:
            self.control.gear = 1  # Forward gear

        self.vehicle.apply_control(self.control)


def tutorial(args):
    """
    This function retrieves front and rear camera data synchronously and displays it in real-time using Pygame.
    Additionally, manual control over the vehicle is enabled.
    """
    # Initialize Pygame
    pygame.init()
    # Create a window to display both front and rear camera images side by side
    display = pygame.display.set_mode((args.width * 2, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

    # Connect to the server
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # Faster refresh for real-time display
    world.apply_settings(settings)

    vehicle = None
    front_camera = None
    rear_camera = None

    try:
        # Search the desired blueprints
        vehicle_bp = bp_lib.filter("vehicle.lincoln.mkz_2017")[0]
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]

        # Configure the front and rear camera blueprints
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))

        # Spawn the vehicle
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[0])

        # Initialize manual control
        manual_control = ManualControl(vehicle)

        # Front camera - placed in front of the vehicle
        front_camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
            attach_to=vehicle)

        # Rear camera - placed at the back of the vehicle
        rear_camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=-1.6, z=1.6), carla.Rotation(yaw=180)),
            attach_to=vehicle)

        # Create queues to store front and rear camera data
        front_image_queue = Queue()
        rear_image_queue = Queue()

        # Set up listeners for both cameras
        front_camera.listen(lambda data: sensor_callback(data, front_image_queue))
        rear_camera.listen(lambda data: sensor_callback(data, rear_image_queue))

        # Main loop for real-time display and manual control
        running = True
        while running:
            world.tick()

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == K_ESCAPE:
                    running = False

            # Parse keyboard events for manual vehicle control
            manual_control.parse_events()

            try:
                # Get camera data from the queues
                front_image_data = front_image_queue.get(True, 1.0)
                rear_image_data = rear_image_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            # Convert the raw front camera image data to a numpy array and display it using Pygame
            front_im_array = np.copy(np.frombuffer(front_image_data.raw_data, dtype=np.dtype("uint8")))
            front_im_array = np.reshape(front_im_array, (front_image_data.height, front_image_data.width, 4))
            front_im_array = front_im_array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB

            # Convert the raw rear camera image data to a numpy array and display it using Pygame
            rear_im_array = np.copy(np.frombuffer(rear_image_data.raw_data, dtype=np.dtype("uint8")))
            rear_im_array = np.reshape(rear_im_array, (rear_image_data.height, rear_image_data.width, 4))
            rear_im_array = rear_im_array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB

            # Create Pygame surfaces from the numpy arrays
            front_surface = pygame.surfarray.make_surface(front_im_array.swapaxes(0, 1))
            rear_surface = pygame.surfarray.make_surface(rear_im_array.swapaxes(0, 1))

            # Display front camera on the left half and rear camera on the right half of the Pygame window
            display.blit(front_surface, (0, 0))
            display.blit(rear_surface, (args.width, 0))
            pygame.display.flip()

    finally:
        # Apply the original settings when exiting
        world.apply_settings(original_settings)

        # Destroy the actors
        if front_camera:
            front_camera.destroy()
        if rear_camera:
            rear_camera.destroy()
        if vehicle:
            vehicle.destroy()

        # Quit Pygame
        pygame.quit()


def main():
    """Start function"""
    argparser = argparse.ArgumentParser(
        description='CARLA Front and Rear Camera real-time visualization using manual control and Pygame')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='680x420',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=500,
        type=int,
        help='number of frames to record (default: 500)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        tutorial(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()


