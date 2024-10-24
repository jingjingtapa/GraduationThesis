import carla
import pygame
import numpy as np
import cv2
import os
import random

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

# BEV 변환 영역 설정 값
wx_min, wx_max, wx_interval = 7, 40, 0.05
wy_min, wy_max, wy_interval = -10, 10, 0.05

# BEV 변환기 초기화
bev = BEVConverter(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval)

# Pygame 초기화
pygame.init()

# Pygame 디스플레이 설정
screen_width = int((wy_max-wy_min)/wy_interval)
screen_height = int((wx_max-wx_min)/wx_interval)*2
screen = pygame.display.set_mode((screen_width, screen_height))

# 카메라 설정
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
    R = np.identity(4)
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

vehicle_physics_control = vehicle.get_physics_control()

for wheel in vehicle_physics_control.wheels:
    wheel.suspension_stiffness = 1000000000.0
    wheel.wheel_damping_rate = 0.00001

vehicle_physics_control.center_of_mass = carla.Vector3D(0.0, 0.0, -0.5)
vehicle.apply_physics_control(vehicle_physics_control)

vehicle.set_autopilot(False)  # Autopilot 비활성화 (manual_control 적용)

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
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = cv2.remap(array, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    array = array[:, :, ::-1]
    if image_type == 'front':
        global image_front
        image_front = array

front_camera.listen(lambda image: process_image(image, 'front'))

# 추가: 교통량(다른 차량들) 스폰 함수
def spawn_traffic(num_vehicles=10):
    traffic_vehicles = []
    for _ in range(num_vehicles):
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True)  # 자율주행 모드 활성화
            traffic_vehicles.append(vehicle)
    return traffic_vehicles

# 교통량 스폰
traffic_vehicles = spawn_traffic(20)  # 교통량을 20대로 설정

running = True
save_image = False

# 이미지 저장 카운터 변수 초기화
cnt = 0

if not os.path.exists('images'):
    os.makedirs('images')

# Manual control 초기화
control = carla.VehicleControl()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_g:
                save_image = True

    # Manual control - 키보드 입력 처리
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        control.throttle = min(control.throttle + 0.05, 1)
    else:
        control.throttle = 0

    if keys[pygame.K_s]:
        control.brake = min(control.brake + 0.05, 1)
    else:
        control.brake = 0

    if keys[pygame.K_a]:
        control.steer = max(control.steer - 0.05, -1)
    elif keys[pygame.K_d]:
        control.steer = min(control.steer + 0.05, 1)
    else:
        control.steer = 0

    # 차량에 control 적용
    vehicle.apply_control(control)

    screen.fill((0, 0, 0))
    if image_front is not None and image_front.size > 0:
        surface = pygame.surfarray.make_surface(image_front.swapaxes(0, 1))
        screen.blit(surface, (0, 0))
        if save_image:
            image_name = f"images/front_view_{cnt}.png"
            pygame.image.save(surface, image_name)
            print(f"Image saved as {image_name}")
            save_image = False
            cnt += 1
    else:
        print("Front view is empty or None")

    pygame.display.flip()

front_camera.stop()

# 종료 시 교통량으로 스폰한 차량들 제거
for traffic_vehicle in traffic_vehicles:
    traffic_vehicle.destroy()

vehicle.destroy()
pygame.quit()


