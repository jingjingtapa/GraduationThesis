import carla
import pygame
import numpy as np
import cv2, sys, os
import BEVConverter

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

    def motion_cancel(self, imu_data):
        roll_rate = imu_data.gyroscope.x
        pitch_rate = imu_data.gyroscope.y
        yaw_rate = imu_data.gyroscope.z
        
       #roll = np.radians(-roll_rate)
        #pitch = np.radians(pitch_rate)
        #yaw = np.radians(-yaw_rate)
        
        R_imu = self.rotation_from_euler(0, pitch_rate, 0)
        T_imu = self.translation_matrix([0, 0, 0])
        motion_cancel_mat = R_imu @ T_imu
        return motion_cancel_mat

    def load_camera_params(self, cal):
        fx, fy = cal['intrinsic']['fx'], cal['intrinsic']['fy']
        u0, v0 = cal['intrinsic']['u0'], cal['intrinsic']['v0']

        pitch, roll, yaw = cal['extrinsic']['pitch'], cal['extrinsic']['roll'], cal['extrinsic']['yaw']
        x, y, z = cal['extrinsic']['x'], cal['extrinsic']['y'], cal['extrinsic']['z']

        baseline = cal['extrinsic']['baseline']
        K = np.array([[fx, 0, u0, 0],
                      [0, fy, v0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        R_veh2cam = np.transpose(self.rotation_from_euler(roll, pitch, yaw))
        T_veh2cam = self.translation_matrix((-x-baseline/2, -y, -z))

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
        bev_image = cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return bev_image


wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = 7, 40, 0.05, -10, 10, 0.05
bev = BEVConverter(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval)

pygame.init()

screen_width = 640 * 2
screen_height = 480 * 2
screen = pygame.display.set_mode((screen_width, screen_height))

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FOV = 90

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
    R[0, 3], R[1, 3], R[2, 3] = -x, -y, -z

    r = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])
    
    RT = r @ R
    return RT

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)

def attach_imu(vehicle, transform):
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_transform = carla.Transform(carla.Location(x=transform[0], y=transform[1], z=transform[2]))
    imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
    return imu

imu_data_global = None

def imu_callback(imu_data):
    global pitch_rate, imu_data_global
    pitch_rate = imu_data.gyroscope.y  
    imu_data_global = imu_data
    bev.motion_cancel(imu_data)

imu_transform = [0.0, 0.0, 1.0]
imu_sensor = attach_imu(vehicle, imu_transform)
imu_sensor.listen(lambda imu_data: imu_callback(imu_data))

def attach_camera(vehicle, transform, fov=FOV):
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{CAMERA_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{CAMERA_HEIGHT}')
    camera_bp.set_attribute('fov', f'{fov}')
    camera_transform = carla.Transform(carla.Location(x=transform[0], y=transform[1], z=transform[2]),
                                       carla.Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5]))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

transform = [2.0, 0.0, 1.5, 0.0, 0.0, 0.0]
front_camera = attach_camera(vehicle, transform)

K = intrinsic(CAMERA_WIDTH, CAMERA_HEIGHT, FOV)
R = extrinsic(transform)

map_x, map_y, bev_height, bev_width = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval,
                                                                           wy_min, wy_max, wy_interval,
                                                                           R, K)

image_front = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)

def process_image(image, image_type):
    global pitch_rate, imu_data_global
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # BGR만 추출
    array = cv2.remap(array, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    # IMU 데이터가 존재할 경우에만 흔들림 보정 적용
    if imu_data_global is not None:
        motion_cancel_mat = bev.motion_cancel(imu_data_global)  # 전역 변수 사용

        # uv_coord에 동차 좌표 추가 (3차원 -> 4차원)
        uv_coord = np.array([array.shape[1] / 2, array.shape[0] / 2, 1, 1])  # 기본 좌표에 1 추가
        uv_coord = motion_cancel_mat @ uv_coord  # 행렬 곱셈

        # 다시 3차원으로 변환 (동차 좌표의 마지막 요소를 제거)
        uv_coord = uv_coord[:3] / uv_coord[3]
    
    array = array[:, :, ::-1]  # BGR -> RGB 변환
    if image_type == 'front':
        global image_front
        image_front = array


front_camera.listen(lambda image: process_image(image, 'front'))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))  
    if image_front is not None and image_front.size > 0:
        surface = pygame.surfarray.make_surface(image_front.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
    else:
        print("Front view is empty or None")

    pygame.display.flip()

front_camera.stop()
imu_sensor.stop()
vehicle.destroy()
pygame.quit()

