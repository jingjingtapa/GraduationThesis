import glob
import os
import sys
import carla
import argparse
from queue import Queue, Empty
import numpy as np
import pygame
import cv2

# Camera sensor callback function to store the data in a queue
def sensor_callback(data, queue):
    queue.put(data)

# Function to compute the intrinsic matrix using image data
def compute_intrinsic_matrix(image_data):
    fov = image_data.fov  # Field of View (from the image data)
    width = image_data.width  # Image width
    height = image_data.height  # Image height

    # Compute focal length from FOV
    focal_length_x = width / (2 * np.tan(np.deg2rad(fov / 2)))
    focal_length_y = height / (2 * np.tan(np.deg2rad(fov / 2)))

    # Construct the intrinsic matrix
    intrinsic_matrix = np.array([[focal_length_x, 0, width / 2],
                                 [0, focal_length_y, height / 2],
                                 [0, 0, 1]])
    return intrinsic_matrix

# Function to compute the extrinsic matrix using the camera actor
def compute_extrinsic_matrix(camera):
    # 카메라의 transform 가져오기
    camera_transform = camera.get_transform()

    # 카메라 위치 (translation)
    translation = np.array([camera_transform.location.x,
                            camera_transform.location.y,
                            camera_transform.location.z])

    # 카메라 회전 (rotation) -> yaw, pitch, roll을 회전 행렬로 변환
    rotation = camera_transform.rotation
    yaw = np.deg2rad(rotation.yaw)
    pitch = np.deg2rad(rotation.pitch)
    roll = np.deg2rad(rotation.roll)

    # Yaw (Z 축 회전)
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw),  0],
                      [0,           0,            1]])

    # Pitch (Y 축 회전)
    R_pitch = np.array([[np.cos(pitch),  0, np.sin(pitch)],
                        [0,              1, 0           ],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    # Roll (X 축 회전)
    R_roll = np.array([[1, 0,           0           ],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])

    # 최종 회전 행렬 = Yaw * Pitch * Roll
    rotation_matrix = np.dot(np.dot(R_yaw, R_pitch), R_roll)

    # Extrinsic matrix 구성
    extrinsic_matrix = np.identity(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = translation

    return extrinsic_matrix

# Class to handle BEV transformation
class BEVConverter:
    def __init__(self, wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval):
        self.wx_min, self.wx_max, self.wx_interval = wx_min, wx_max, wx_interval
        self.wy_min, self.wy_max, self.wy_interval = wy_min, wy_max, wy_interval
        self.map_x, self.map_y = None, None  # 매핑을 캐싱하기 위한 변수 추가

    def generate_direct_backward_mapping(self, extrinsic, intrinsic):
        if self.map_x is not None and self.map_y is not None:
            return self.map_x, self.map_y  # 캐싱된 매핑 반환

        # 기존 BEV 좌표 변환 매핑 계산 (캐싱)
        world_x_coords = np.arange(self.wx_max, self.wx_min, -self.wx_interval)
        world_y_coords = np.arange(self.wy_max, self.wy_min, -self.wy_interval)

        output_height = len(world_x_coords)
        output_width = len(world_y_coords)

        self.map_x = np.zeros((output_height, output_width), dtype=np.float32)
        self.map_y = np.zeros((output_height, output_width), dtype=np.float32)

        for i, world_x in enumerate(world_x_coords):
            for j, world_y in enumerate(world_y_coords):
                world_coord = np.array([world_x, world_y, 0, 1])
                camera_coord = np.dot(extrinsic, world_coord)

                if camera_coord[2] > 1e-6:  # Avoid divide by zero or very small values
                    uv_coord = np.dot(intrinsic[:3, :3], camera_coord[:3])
                    uv_coord /= uv_coord[2]
                    self.map_x[i, j] = uv_coord[0]
                    self.map_y[i, j] = uv_coord[1]
                else:
                    self.map_x[i, j] = -1
                    self.map_y[i, j] = -1

        return self.map_x, self.map_y


    def apply_bev_transform(self, image, extrinsic, intrinsic):
        map_x, map_y = self.generate_direct_backward_mapping(extrinsic, intrinsic)
        bev_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        return bev_image

# Main function to handle the real-time visualization
def tutorial(args):
    pygame.init()
    display = pygame.display.set_mode((args.width * 2, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    vehicle = None
    front_camera = None
    rear_camera = None

    try:
        # Set up vehicle and cameras
        vehicle_bp = bp_lib.filter("vehicle.lincoln.mkz_2017")[0]
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))
        camera_bp.set_attribute("sensor_tick", "0.2")  # 초당 5프레임으로 제한


        vehicle = world.spawn_actor(vehicle_bp, transform=world.get_map().get_spawn_points()[0])
        vehicle.set_autopilot(True)

        # Front camera
        front_camera = world.spawn_actor(camera_bp, transform=carla.Transform(carla.Location(x=1.6, z=1.6)), attach_to=vehicle)

        # Rear camera
        rear_camera = world.spawn_actor(camera_bp, transform=carla.Transform(carla.Location(x=-1.6, z=1.6), carla.Rotation(yaw=180)), attach_to=vehicle)

        # Create queues for storing camera data
        front_image_queue = Queue()
        rear_image_queue = Queue()

        front_camera.listen(lambda data: sensor_callback(data, front_image_queue))
        rear_camera.listen(lambda data: sensor_callback(data, rear_image_queue))

        # Initialize BEV converter
        bev_converter = BEVConverter(wx_min=-20, wx_max=20, wx_interval=0.1, wy_min=-10, wy_max=10, wy_interval=0.1)

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
                print("[Warning] Some sensor data has been missed")
                continue

            # Convert camera data to numpy arrays
            front_image = np.reshape(np.copy(np.frombuffer(front_image_data.raw_data, dtype=np.uint8)),
                                     (front_image_data.height, front_image_data.width, 4))[:, :, :3]
            rear_image = np.reshape(np.copy(np.frombuffer(rear_image_data.raw_data, dtype=np.uint8)),
                                    (rear_image_data.height, rear_image_data.width, 4))[:, :, :3]

            # Now that we have the image data, compute intrinsic matrices
            intrinsic_front = compute_intrinsic_matrix(front_image_data)
            extrinsic_front = compute_extrinsic_matrix(front_camera)

            intrinsic_rear = compute_intrinsic_matrix(rear_image_data)
            extrinsic_rear = compute_extrinsic_matrix(rear_camera)

            # Apply BEV transformation
            bev_front_image = bev_converter.apply_bev_transform(front_image, extrinsic_front, intrinsic_front)
            bev_rear_image = bev_converter.apply_bev_transform(rear_image, extrinsic_rear, intrinsic_rear)

            # Convert BEV image to Pygame surfaces
            bev_front_surface = pygame.surfarray.make_surface(bev_front_image.swapaxes(0, 1))
            bev_rear_surface = pygame.surfarray.make_surface(bev_rear_image.swapaxes(0, 1))

            # Display the BEV-transformed images
            display.blit(bev_front_surface, (0, 0))
            display.blit(bev_rear_surface, (args.width, 0))
            pygame.display.flip()

    finally:
        if front_camera:
            front_camera.destroy()
        if rear_camera:
            rear_camera.destroy()
        if vehicle:
            vehicle.destroy()

        pygame.quit()

# Main function entry point
def main():
    argparser = argparse.ArgumentParser(description='CARLA Front and Rear Camera BEV real-time visualization using Pygame')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='680x420', help='window resolution (default: 480x270)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        tutorial(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()






