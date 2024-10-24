import carla
import pygame
import numpy as np
import cv2

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


# 카메라 파라미터 설정
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

    R = np.identity(4)
    R[0, 3], R[1, 3], R[2, 3] = -x, -y, -z

    r = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])
    RT = r @ R
    return RT


# Pygame 초기화
pygame.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Carla 시뮬레이터와 연결
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Ego 차량 스폰 및 카메라 부착
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 차량의 물리 파라미터 조정
vehicle_physics_control = vehicle.get_physics_control()
for wheel in vehicle_physics_control.wheels:
    wheel.suspension_stiffness = 100000.0  # 매우 높은 값으로 설정
    wheel.wheel_damping_rate = 1000.0  # 댐퍼 값도 높게 설정
vehicle.apply_physics_control(vehicle_physics_control)

vehicle.set_autopilot(True)

# IMU 센서 부착 함수
def attach_imu_sensor(vehicle):
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_transform = carla.Transform(carla.Location(x=0, y=0, z=1))
    imu_sensor = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
    return imu_sensor

# IMU 센서 부착
imu_sensor = attach_imu_sensor(vehicle)

# 카메라 부착 함수 정의
def attach_camera(vehicle, transform, fov=FOV):
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{CAMERA_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{CAMERA_HEIGHT}')
    camera_bp.set_attribute('fov', f'{fov}')
    camera_transform = carla.Transform(carla.Location(x=transform[0], y=transform[1], z=transform[2]),
                                       carla.Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5]))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

# 카메라 부착
transform = [2.0, 0.0, 1.5, 0.0, 0.0, 0.0]
front_camera = attach_camera(vehicle, transform)

K = intrinsic(CAMERA_WIDTH, CAMERA_HEIGHT, FOV)
R = extrinsic(transform)

bev = BEVConverter(7, 40, 0.05, -10, 10, 0.05)
map_x, map_y, bev_height, bev_width = bev.generate_direct_backward_mapping(7, 40, 0.05, -10, 10, 0.05, R, K)

# IMU 데이터를 기반으로 카메라 흔들림 보정
def stabilize_camera(imu_data, camera):
    roll = imu_data.gyroscope.x
    pitch = imu_data.gyroscope.y
    yaw = imu_data.gyroscope.z
    
    # 카메라 회전 보정 (피칭, 롤링, 요잉 반대 방향으로 보정)
    camera_transform = camera.get_transform()
    camera_transform.rotation.pitch += pitch  # 피칭 보정
    camera_transform.rotation.roll += roll    # 롤링 보정
    camera_transform.rotation.yaw += yaw      # 요잉 보정
    
    camera.set_transform(camera_transform)

# 카메라에서 받은 데이터를 처리하는 배열
image_front = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)

# 콜백 함수: 카메라로부터 받은 이미지 처리
def process_image(image, image_type):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # BGR만 추출
    array = cv2.remap(array, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    array = array[:, :, ::-1]  # BGR -> RGB 변환
    if image_type == 'front':
        global image_front
        image_front = array
        
# 콜백 함수 등록
front_camera.listen(lambda image: process_image(image, 'front'))        
   
imu_data = None
def imu_callback(data):
    global imu_data
    imu_data = data

imu_sensor.listen(imu_callback)

# Pygame을 통한 실시간 시각화 루프
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    if imu_data is not None:
        stabilize_camera(imu_data, front_camera)  # 카메라 흔들림 보정
   
    
    # 화면 배치 설정
    screen.fill((0, 0, 0))  # 화면 초기화
    if image_front is not None and image_front.size > 0:
        surface = pygame.surfarray.make_surface(image_front.swapaxes(0, 1))
        screen.blit(surface, (0, 0))

    pygame.display.flip()

# 종료 시 actor 정리
front_camera.stop()
imu_sensor.stop()
vehicle.destroy()
pygame.quit()

