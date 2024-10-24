import carla
import pygame
import numpy as np
import cv2
import os
import random
import time

class BEVConverter:
    def __init__(self, wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval):
        self.wx_min, self.wx_max, self.wx_interval, self.wy_min, self.wy_max, self.wy_interval = wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval

    def rotation_from_euler(self, roll=1., pitch=1., yaw=1.):
        si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
        ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        R = np.identity(4)
        R[0, 0] = cj * ck
        R[0, 1] = sj * sc - cs
        R[0, 2] = sj * cc + ss
        R[1, 0] = cj * sk
        R[1, 1] = sj * ss + cc
        R[1, 2] = sj * cs - sc
        R[2, 0] = -sj
        R[2, 1] = cj * si
        R[2, 2] = cj * ci
        return R

    def translation_matrix(self, vector):
        M = np.identity(4)
        M[:3, 3] = vector[:3]
        return M

    def motion_cancel(self, cal):
        imu = np.array(cal['sensor']['sensor_T_ISO_8855'])  # 3*4
        rotation, translation = imu[:2,:2].T, -imu[:2,3]
        
        motion_cancel_mat = np.identity(3)
        motion_cancel_mat[:2,:2] = rotation
        motion_cancel_mat[:2,2] = translation

        return motion_cancel_mat

    def load_camera_params(self, cal):  #c: calibration
        fx, fy = cal['intrinsic']['fx'], cal['intrinsic']['fy']
        u0, v0 = cal['intrinsic']['u0'], cal['intrinsic']['v0']

        pitch, roll, yaw = cal['extrinsic']['pitch'], cal['extrinsic']['roll'], cal['extrinsic']['yaw']
        x, y, z = cal['extrinsic']['x'], cal['extrinsic']['y'], cal['extrinsic']['z']

        baseline = cal['extrinsic']['baseline']
        # Intrinsic
        K = np.array([[fx, 0, u0, 0],
                    [0, fy, v0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

        # Extrinsic
        R_veh2cam = np.transpose(self.rotation_from_euler(roll, pitch, yaw))
        T_veh2cam = self.translation_matrix((-x-baseline/2, -y, -z))

        # Rotate to camera coordinates
        R = np.array([[0., -1., 0., 0.],
                    [0., 0., -1., 0.],
                    [1., 0., 0., 0.],
                    [0., 0., 0., 1.]])

        RT = R @ R_veh2cam @ T_veh2cam
        return RT, K

    def generate_direct_backward_mapping(self,
        world_x_min, world_x_max, world_x_interval, 
        world_y_min, world_y_max, world_y_interval, 
        extrinsic, intrinsic):
        
        world_x_coords = np.arange(world_x_max, world_x_min, -world_x_interval)
        world_y_coords = np.arange(world_y_max, world_y_min, -world_y_interval)
        
        output_height = len(world_x_coords)
        output_width = len(world_y_coords)
        
        map_x = np.zeros((output_height, output_width)).astype(np.float32)
        map_y = np.zeros((output_height, output_width)).astype(np.float32)
    
        
        world_z = 0
        for i, world_x in enumerate(world_x_coords):
            for j, world_y in enumerate(world_y_coords):
                world_coord = [world_x, world_y, world_z, 1]
                camera_coord = extrinsic[:3, :] @ world_coord
                uv_coord = intrinsic[:3, :3] @ camera_coord
                uv_coord /= uv_coord[2]

                map_x[i][j] = uv_coord[0]
                map_y[i][j] = uv_coord[1]
                
        return map_x, map_y, output_height, output_width
        
    def get_BEV_image(self, image, map_x, map_y):
        bev_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return bev_image

def get_bounding_box_world_coords(vehicle):
    bbox = vehicle.bounding_box
    vertices = np.array([[-bbox.extent.x, -bbox.extent.y, -bbox.extent.z],
                         [-bbox.extent.x,  bbox.extent.y, -bbox.extent.z],
                         [ bbox.extent.x,  bbox.extent.y, -bbox.extent.z],
                         [ bbox.extent.x, -bbox.extent.y, -bbox.extent.z],
                         [-bbox.extent.x, -bbox.extent.y,  bbox.extent.z],
                         [-bbox.extent.x,  bbox.extent.y,  bbox.extent.z],
                         [ bbox.extent.x,  bbox.extent.y,  bbox.extent.z],
                         [ bbox.extent.x, -bbox.extent.y,  bbox.extent.z]])

    vehicle_transform = vehicle.get_transform()
    vehicle_matrix = np.array(vehicle_transform.get_matrix())

    vertices_world = []
    for vertex in vertices:
        vertex_homogeneous = np.append(vertex, 1)
        world_coords = vehicle_matrix @ vertex_homogeneous
        vertices_world.append(world_coords[:3])
    return np.array(vertices_world)

def transform_to_ego_coords(vertices_world, ego_inverse_matrix):
    vertices_ego = []
    for vertex in vertices_world:
        vertex_homogeneous = np.append(vertex, 1)
        ego_coords = ego_inverse_matrix @ vertex_homogeneous
        vertices_ego.append(ego_coords[:3])
    return np.array(vertices_ego)

def intrinsic(camera_width, camera_height, fov):
    f_x = camera_width / (2 * np.tan(np.radians(fov / 2)))
    f_y = f_x
    c_x, c_y = camera_width / 2, camera_height / 2
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    return K

def extrinsic(t):
    x, y, z = t[0], t[1], t[2]
    roll, pitch, yaw = t[3], t[4], t[5]
    R = np.identity(4)
    R[0, 3], R[1, 3], R[2, 3] = -x, -y, -z
    r = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])
    RT = r @ R
    return RT

def attach_camera(vehicle, transform, fov=90):
    camera_bp = vehicle.get_world().get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', str(fov))
    camera_transform = carla.Transform(carla.Location(x=transform[0], y=transform[1], z=transform[2]),
                                       carla.Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5]))
    camera = vehicle.get_world().spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

def spawn_traffic(world, blueprint_library, num_vehicles=10):
    traffic_vehicles = []
    for _ in range(num_vehicles):
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True)
            traffic_vehicles.append(vehicle)
    return traffic_vehicles

# Main Pygame loop with BEV visualization
def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    ego_vehicle_bp = blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)

    traffic_vehicles = spawn_traffic(world, blueprint_library, 20)

    bev = BEVConverter(7, 40, 0.05, -10, 10, 0.05)
    transform = [2.0, 0.0, 1.5, 0.0, 0.0, 0.0]
    front_camera = attach_camera(ego_vehicle, transform)

    K = intrinsic(640, 480, 90)
    R = extrinsic(transform)

    map_x, map_y, bev_height, bev_width = bev.generate_direct_backward_mapping(7, 40, 0.05, -10, 10, 0.05, R, K)

    image_front = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)

    front_camera.listen(lambda image: process_image(image, 'front'))

    def process_image(image, image_type):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = cv2.remap(array, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        array = array[:, :, ::-1]
        global image_front
        image_front = array

    running = True
    save_image = False
    cnt = 0
    ego_inverse_matrix = np.linalg.inv(np.array(ego_vehicle.get_transform().get_matrix()))

    if not os.path.exists('images'):
        os.makedirs('images')

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    save_image = True

        vehicles = world.get_actors().filter('vehicle.*')
        ego_location = ego_vehicle.get_location()

        screen.fill((0, 0, 0))
        if image_front is not None and image_front.size > 0:
            surface = pygame.surfarray.make_surface(image_front.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            if save_image:
                image_name = f"images/front_view_{cnt}.png"
                pygame.image.save(surface, image_name)

                with open(f"images/bbox_{cnt}.txt", "w") as bbox_file:
                    for vehicle in vehicles:
                        if vehicle.id != ego_vehicle.id:
                            distance = ego_location.distance(vehicle.get_location())
                            if distance <= 20:
                                bbox_world_coords = get_bounding_box_world_coords(vehicle)
                                bbox_ego_coords = transform_to_ego_coords(bbox_world_coords, ego_inverse_matrix)
                                bbox_file.write(f"Vehicle {vehicle.type_id} Bounding Box in Ego Coordinates: {bbox_ego_coords.tolist()}\n")
                print(f"Image and bounding boxes saved as {image_name} and bbox_{cnt}.txt")
                save_image = False
                cnt += 1
        else:
            print("Front view is empty or None")

        pygame.display.flip()

    front_camera.stop()
    for vehicle in traffic_vehicles:
        vehicle.destroy()

    ego_vehicle.destroy()
    pygame.quit()

if __name__ == '__main__':
    main()

