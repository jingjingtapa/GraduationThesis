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


wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = 7, 40, 0.05, -10, 10, 0.05

bev = BEVConverter(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval)

# Pygame 초기화
pygame.init()

# Pygame 디스플레이 설정
screen_width = int((wy_max-wy_min)/wy_interval)
screen_height = int((wx_max-wx_min)/wx_interval)*2  # 전방과 후방 카메라 영상을 합쳐서 표시하기 위해 높이를 2배로 설정
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
    
    si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
    ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    R = np.identity(4)
    R[0, 3], R[1, 3], R[2, 3] = -x, -y, -z

    r = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])
    
    RT = r @ R
    return RT

# Carla 시뮬레이터와 연결
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Ego 차량 스폰 및 카메라 부착
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 차량의 물리 파라미터 설정을 가져오기
vehicle_physics_control = vehicle.get_physics_control()

# 서스펜션 강도와 댐퍼 값을 조정하여 피칭을 줄임 -> 극단적으로 설정
for wheel in vehicle_physics_control.wheels:
    wheel.suspension_stiffness = 1000000000.0  # 서스펜션을 더 강하게 설정
    wheel.wheel_damping_rate = 0.00001 # 댐퍼 값을 낮게 설정하여 흔들림 줄이기

# 차량의 질량 중심을 낮게 설정하여 피칭을 줄임
vehicle_physics_control.center_of_mass = carla.Vector3D(0.0, 0.0, -0.5)  # Z축을 낮게 설정하여 무게중심을 아래로 내림

# 새로운 물리 파라미터를 차량에 적용
vehicle.apply_physics_control(vehicle_physics_control)

# 차량 자동 운전 설정
vehicle.set_autopilot(True)

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

# 전방 카메라 부착
front_transform = [2.0, 0.0, 1.5, 0.0, 0.0, 0.0]
front_camera = attach_camera(vehicle, front_transform)

# 후방 카메라 부착 (yaw 각도를 180도로 설정하여 후방을 바라보도록 설정)
rear_transform = [-2.0, 0.0, 1.5, 0.0, 180.0, 0.0]
rear_camera = attach_camera(vehicle, rear_transform)

K_front = intrinsic(CAMERA_WIDTH, CAMERA_HEIGHT, FOV)
R_front = extrinsic(front_transform)

K_rear = intrinsic(CAMERA_WIDTH, CAMERA_HEIGHT, FOV)
R_rear = extrinsic(rear_transform)

map_x_front, map_y_front, bev_height, bev_width = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval,
                                                                                      wy_min, wy_max, wy_interval,
                                                                                      R_front, K_front)

# 후방 카메라 BEV 변환에서도 전방과 동일한 크기로 설정
map_x_rear, map_y_rear, bev_height_rear, bev_width_rear = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval,
                                                                                               wy_min, wy_max, wy_interval,
                                                                                               R_rear, K_rear)
                                                                                               
screen_width = bev_width
screen_height = bev_height * 2                                                                                                


# 각 카메라에서 받은 데이터를 화면에 그리기 위한 배열
image_front = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
image_rear = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)

# 콜백 함수: 카메라로부터 받은 이미지 처리
def process_image(image, image_type):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # BGR만 추출
    array = array[:, :, ::-1]  # BGR -> RGB 변환
    if image_type == 'front':
        global image_front
        array = cv2.remap(array, map_x_front, map_y_front, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        image_front = array
    elif image_type == 'rear':
        global image_rear
        array = cv2.remap(array, map_x_rear, map_y_rear, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        image_rear = array

# 콜백 함수 등록
front_camera.listen(lambda image: process_image(image, 'front'))
# 후방 카메라 콜백 등록
rear_camera.listen(lambda image: process_image(image, 'rear'))

# Pygame을 통한 실시간 시각화 루프
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 화면 초기화
    screen.fill((0, 0, 0))

    # 전방 카메라 영상 시각화 (화면 위쪽에 배치)
    if image_front is not None and image_front.size > 0:
        surface_front = pygame.surfarray.make_surface(image_front.swapaxes(0, 1))
        screen.blit(surface_front, (0, 0))

    # 후방 카메라 영상 시각화 (화면 아래쪽에 배치)
    if image_rear is not None and image_rear.size > 0:
        surface_rear = pygame.surfarray.make_surface(image_rear.swapaxes(0, 1))
        screen.blit(surface_rear, (0, bev_height))

    # 화면 업데이트
    pygame.display.flip()

# 종료 시 actor 정리
front_camera.stop()
rear_camera.stop()
vehicle.destroy()
pygame.quit()


