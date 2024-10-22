#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Real-time visualization of front and rear cameras using Pygame with image saving functionality for 10 seconds
"""

import glob
import os
import sys
import time

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


def sensor_callback(data, queue):
    """
    This simple callback stores the camera data in a thread-safe Python Queue
    to be retrieved from the main thread.
    """
    queue.put(data)


def tutorial(args):
    """
    This function retrieves front and rear camera data synchronously and displays it in real-time using Pygame.
    It also saves the images for 10 seconds.
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

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

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
        vehicle.set_autopilot(True)

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

        # Variables for saving images
        save_images = True
        save_duration = 10  # seconds
        start_time = time.time()
        frame_count = 0

        # Main loop for real-time display
        running = True
        while running:
            world.tick()

            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

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

            # Save images for 10 seconds
            if save_images and time.time() - start_time < save_duration:
                front_image_filename = f"front_camera_frame_{frame_count}.png"
                rear_image_filename = f"rear_camera_frame_{frame_count}.png"
                pygame.image.save(front_surface, front_image_filename)
                pygame.image.save(rear_surface, rear_image_filename)
                frame_count += 1

            if time.time() - start_time >= save_duration:
                save_images = False  # Stop saving after 10 seconds

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
        description='CARLA Front and Rear Camera real-time visualization using Pygame with image saving functionality')
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

