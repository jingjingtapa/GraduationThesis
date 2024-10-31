import numpy as np
import cv2, random, carla, pygame, getpass, os, time, math
from PIL import Image

num_vehicles = 40
sampling_time = 0.1 # tick : 20Hz
scene_delay = 10

class BEVConverter:
    def __init__(self, wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval):
        self.wx_min, self.wx_max, self.wx_interval, self.wy_min, self.wy_max, self.wy_interval = wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval
        self.CAMERA_WIDTH, self.CAMERA_HEIGHT, self.FOV = 640, 480, 90

    def intrinsic(self, camera_width, camera_height, fov):
        f_x = camera_width / (2 * np.tan(np.radians(fov / 2)))
        f_y = f_x
        c_x, c_y = camera_width / 2, camera_height / 2

        K = np.array([[f_x, 0, c_x],
                    [0, f_y, c_y],
                    [0, 0, 1]])
        return K

    def extrinsic(self, t):
        x, y, z = t[0], t[1], t[2]
        pitch, yaw, roll = np.deg2rad(t[3]), np.deg2rad(t[4]), np.deg2rad(t[5])
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

        T = np.identity(4)
        T[0, 3], T[1, 3], T[2, 3] = -x, -y, -z

        r = np.array([[0., -1., 0., 0.],
                    [0., 0., -1., 0.],
                    [1., 0., 0., 0.],
                    [0., 0., 0., 1.]])
        
        RT = r @ R @ T
        return RT
    
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
                # [world_x, world_y, 0, 1] -> [u, v, 1]
                world_coord = [world_x, world_y, world_z, 1]
                camera_coord = extrinsic[:3, :] @ world_coord # 3*4 * 4*1 = 3*1
                uv_coord = intrinsic[:3, :3] @ camera_coord # 3*3 * 3*1 = 3*1
                uv_coord /= uv_coord[2]

                map_x[i][j] = uv_coord[0]
                map_y[i][j] = uv_coord[1]
                
        return map_x, map_y, output_height, output_width
    
def spawn_vehicles(world, blueprint_library, num_vehicles):
    spawn_points = world.get_map().get_spawn_points()
    vehicles_list = []

    emergency_vehicles = ['vehicle.*firetruck', 'vehicle.*ambulance']
    for emergency in emergency_vehicles:
        vehicle_bp = random.choice(blueprint_library.filter(emergency))
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True)
            vehicles_list.append(vehicle)

    for i in range(num_vehicles - len(emergency_vehicles)):
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True)
            vehicles_list.append(vehicle)

    return vehicles_list

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = sampling_time
world.apply_settings(settings)

actors = world.get_actors().filter('vehicle.*')
for actor in actors:
    actor.destroy()

# Ego 차량 스폰 및 카메라 부착
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_point = world.get_map().get_spawn_points()[0]
ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
ego_vehicle.set_autopilot(True)

physics_control = ego_vehicle.get_physics_control()
physics_control.center_of_mass.z = -0.5 # 무게 중심 낮추기
for wheel in physics_control.wheels: # 서스펜션 강도 및 댐핑 조정, 타이어 마찰 계수 조정
    wheel.suspension_stiffness = 10000.0
    wheel.damping_rate = 0.01
    wheel.tire_friction = 3.0

ego_vehicle.apply_physics_control(physics_control) # 변경된 물리 속성 적용
length, width = ego_vehicle.bounding_box.extent.x*2, ego_vehicle.bounding_box.extent.y*2

traffic_vehicles = spawn_vehicles(world, blueprint_library, num_vehicles)

wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = int(length/2), 30, 0.05, -10, 10, 0.05
bev = BEVConverter(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval)

pygame.init()
screen_width = int((wy_max - wy_min) / wy_interval)
screen_height = int(((wx_max - wx_min) / wx_interval)*2 + wx_min/wx_interval * 2)
screen = pygame.display.set_mode((screen_width, screen_height))

def attach_camera(vehicle, transform, sampling_time):
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{bev.CAMERA_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{bev.CAMERA_HEIGHT}')
    # camera_bp.set_attribute('sensor_tick', f'{sampling_time}')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=transform[0], y=transform[1], z=transform[2]),
                                       carla.Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5]))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

# 카메라 부착 (전방 카메라)
cz = 2.5
c_pitch = -30.0
front_transform = [0.0, 0.0, cz, c_pitch, 0.0, 0.0]
rear_transform = [0.0, 0.0, cz, c_pitch, 180.0, 0.0]
left_transform = [0.0, 0.0, cz, c_pitch, -90.0, 0.0]
right_transform = [0.0, 0.0, cz, c_pitch, 90.0, 0.0]

front_camera = attach_camera(ego_vehicle, front_transform,sampling_time)
rear_camera = attach_camera(ego_vehicle, rear_transform,sampling_time)
left_camera = attach_camera(ego_vehicle,left_transform,sampling_time)
right_camera = attach_camera(ego_vehicle, right_transform,sampling_time)

K = bev.intrinsic(bev.CAMERA_WIDTH, bev.CAMERA_HEIGHT, bev.FOV)
R = bev.extrinsic(front_transform)
# R_left = bev.extrinsic(left_transform)

map_x_front, map_y_front, bev_height, bev_width = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval,R, K)
map_x_rear, map_y_rear, _, _ = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval, R, K)

map_x_left, map_y_left, bev_left_height, bev_left_width = bev.generate_direct_backward_mapping(math.trunc(width/2), wy_max, wy_interval, -10, 10, wx_interval, R, K)
map_x_right, map_y_right, _, _ = bev.generate_direct_backward_mapping(math.trunc(width/2), wy_max, wy_interval, -10, 10, wx_interval, R, K)

image_front = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
image_rear = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
image_left = np.zeros((bev_left_width, bev_left_height, 3), dtype=np.uint8)
image_right = np.zeros((bev_left_width, bev_left_height, 3), dtype=np.uint8)

def process_image(image, image_type):
    array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3] # BGR
    bev_image = None
    if image_type == 'front':
        bev_image = cv2.remap(array, map_x_front, map_y_front, cv2.INTER_NEAREST)
    elif image_type == 'rear':
        bev_image = cv2.remap(array, map_x_rear, map_y_rear, cv2.INTER_NEAREST)
    elif image_type == 'left':
        bev_image = cv2.rotate(cv2.remap(array, map_x_left, map_y_left, cv2.INTER_NEAREST), cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif image_type == 'right':
        bev_image = cv2.rotate(cv2.remap(array, map_x_right, map_y_right, cv2.INTER_NEAREST), cv2.ROTATE_90_CLOCKWISE)
    
    if bev_image is not None:
        globals()[f"image_{image_type}"] = cv2.cvtColor(bev_image, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환

def convert_black_to_transparent(cv_image):
    pil_image = Image.fromarray(cv_image).convert('RGBA') # RGB -> RGBA
    datas = pil_image.getdata()
    new_data = []
    for item in datas:
        if item[:3] == (0,0,0):
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)
    
    pil_image.putdata(new_data)
    return pil_image

front_camera.listen(lambda image: process_image(image, 'front'))
rear_camera.listen(lambda image: process_image(image, 'rear'))
left_camera.listen(lambda image: process_image(image, 'left'))
right_camera.listen(lambda image: process_image(image, 'right'))

def get_bounding_box_world_coords(vehicle):
    bbox = vehicle.bounding_box
    vertices = np.array([[-bbox.extent.x, -bbox.extent.y, -bbox.extent.z],
                         [-bbox.extent.x,  bbox.extent.y, -bbox.extent.z],
                         [ bbox.extent.x,  bbox.extent.y, -bbox.extent.z],
                         [ bbox.extent.x, -bbox.extent.y, -bbox.extent.z]])

    vehicle_transform = vehicle.get_transform()
    vehicle_matrix = np.array(vehicle_transform.get_matrix())

    vertices_world = []
    for vertex in vertices:
        vertex_homogeneous = np.append(vertex, 1)  # 동차 좌표계 변환
        world_coords = vehicle_matrix @ vertex_homogeneous  # 월드 좌표계로 변환
        vertices_world.append(world_coords[:3])  # 3D 좌표로 변환
    return np.array(vertices_world)

def transform_to_ego_coords(vertices_world, ego_inverse_matrix):
    vertices_ego = []
    for vertex in vertices_world:
        vertex_homogeneous = np.append(vertex, 1)  # 동차 좌표계 변환
        ego_coords = ego_inverse_matrix @ vertex_homogeneous  # Ego 좌표계로 변환
        vertices_ego.append(ego_coords[:3])  # 3D 좌표로 변환
    return np.array(vertices_ego)

def ego_to_bev_coords(x, y, wx_max, wy_min, wx_interval, wy_interval):
    bev_x = int((y - wy_min) / wy_interval)  # y -> BEV 이미지의 x축
    bev_y = int((wx_max - x) / wx_interval)  # x -> BEV 이미지의 y축
    return bev_x, bev_y

splits = ['train','val','test']
num_splits = [100,10,20]
conditions = ['morning_normal','morning_foggy','night_normal','night_foggy']
# conditions = ['night_foggy']
num_img = 30
def set_condition(condition):
    if condition == 'morning_normal':
        weather = carla.WeatherParameters(
            cloudiness=0.0,      # 구름의 양 (0.0 - 100.0)
            precipitation=0.0,   # 강수량 (0.0 - 100.0)
            precipitation_deposits=0.0,  # 젖은 정도 (0.0 - 100.0)
            sun_altitude_angle=90.0,  # 태양의 고도 (0.0 - 90.0) - 밤/낮
            fog_density=0.0,         # 안개의 밀도 (0.0 - 100.0)
            fog_distance=15.0,        # 안개가 보이는 거리 (짧을수록 짙은 안개 느낌)
            fog_falloff=0.0           # 안개가 퍼지는 정도 (0.0 - 5.0)
        )
    elif condition == 'night_normal':
        weather = carla.WeatherParameters(
            cloudiness=0.0,      # 구름의 양 (0.0 - 100.0)
            precipitation=0.0,   # 강수량 (0.0 - 100.0)
            precipitation_deposits=0.0,  # 젖은 정도 (0.0 - 100.0)
            sun_altitude_angle=0.0,  # 태양의 고도 (0.0 - 90.0) - 밤/낮
            fog_density=0.0,         # 안개의 밀도 (0.0 - 100.0)
            fog_distance=15.0,        # 안개가 보이는 거리 (짧을수록 짙은 안개 느낌)
            fog_falloff=0.0           # 안개가 퍼지는 정도 (0.0 - 5.0)
        )
    elif condition == 'morning_foggy':
        weather = carla.WeatherParameters(
            cloudiness=0.0,      # 구름의 양 (0.0 - 100.0)
            precipitation=80.0,   # 강수량 (0.0 - 100.0)
            precipitation_deposits=0.0,  # 젖은 정도 (0.0 - 100.0)
            sun_altitude_angle=90.0,  # 태양의 고도 (0.0 - 90.0) - 밤/낮
            fog_density=80.0,         # 안개의 밀도 (0.0 - 100.0)
            fog_distance=5.0,        # 안개가 보이는 거리 (짧을수록 짙은 안개 느낌)
            fog_falloff=0.0           # 안개가 퍼지는 정도 (0.0 - 5.0)
        )
    elif condition == 'night_foggy':
        weather = carla.WeatherParameters(
            cloudiness=0.0,      # 구름의 양 (0.0 - 100.0)
            precipitation=80.0,   # 강수량 (0.0 - 100.0)
            precipitation_deposits=0.0,  # 젖은 정도 (0.0 - 100.0)
            sun_altitude_angle=0.0,  # 태양의 고도 (0.0 - 90.0) - 밤/낮
            fog_density=80.0,         # 안개의 밀도 (0.0 - 100.0)
            fog_distance=5.0,        # 안개가 보이는 거리 (짧을수록 짙은 안개 느낌)
            fog_falloff=0.0           # 안개가 퍼지는 정도 (0.0 - 5.0)
        )
    world.set_weather(weather)

running = True
username = getpass.getuser()
for j in range(len(conditions)):
    for i in range(len(splits)):
        condition ,split = conditions[j],splits[i]
        set_condition(condition)

        dir_download = f'/home/{username}/다운로드/dataset'
        if not os.path.isdir(f'{dir_download}/{split}/images/{condition}'):
            os.makedirs(f'{dir_download}/{split}/images/{condition}')

        if not os.path.isdir(f'{dir_download}/{split}/labels/{condition}'):
            os.makedirs(f'{dir_download}/{split}/labels/{condition}')

        def save(screen, traffic_list, cnt_scene, cnt_img):
            dir_scene_img = f'{dir_download}/{split}/images/{condition}/{cnt_scene}'
            dir_scene_lbl = f'{dir_download}/{split}/labels/{condition}/{cnt_scene}'
            if not os.path.isdir(dir_scene_img): os.makedirs(dir_scene_img)
            if not os.path.isdir(dir_scene_lbl): os.makedirs(dir_scene_lbl)
            pygame.image.save(screen, f'{dir_scene_img}/{cnt_scene}_{cnt_img}.png') # 이미지 저장
            label_filename = f'{dir_scene_lbl}/{cnt_scene}_{cnt_img}.txt' # 레이블 저장
            with open(label_filename, 'w') as file:
                for bbox in traffic_list:
                    file.write(" ".join(map(str, bbox)) + "\n")
        
        cnt_scene, cnt_img = 0, 1
        while running and cnt_scene <= num_splits[i]:
            world.tick()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            screen.fill((0, 0, 0))
            
            front_pil = convert_black_to_transparent(image_front)
            rear_pil = convert_black_to_transparent(cv2.flip(image_rear,-1))
            left_pil = convert_black_to_transparent(image_left)
            right_pil = convert_black_to_transparent(image_right)

            vertical_padding = Image.new("RGBA", (front_pil.width, int(wx_min / wx_interval * 2)), (0, 0, 0, 0))
            front_rear_combined = Image.new("RGBA", (front_pil.width, front_pil.height + rear_pil.height + vertical_padding.height), (0, 0, 0, 0))
            front_rear_combined.paste(front_pil, (0, 0))
            front_rear_combined.paste(vertical_padding, (0, front_pil.height))
            front_rear_combined.paste(rear_pil, (0, front_pil.height + vertical_padding.height))

            # 좌우 이미지 결합
            horizontal_padding = Image.new("RGBA", (0, left_pil.height), (0, 0, 0, 0))
            left_right_combined = Image.new("RGBA", (left_pil.width + right_pil.width + horizontal_padding.width, left_pil.height), (0, 0, 0, 0))
            left_right_combined.paste(left_pil, (0, 0))
            left_right_combined.paste(horizontal_padding, (left_pil.width, 0))
            left_right_combined.paste(right_pil, (left_pil.width + horizontal_padding.width, 0))

            front_rear_surface = pygame.image.fromstring(front_rear_combined.tobytes(), front_rear_combined.size, "RGBA")
            left_right_surface = pygame.image.fromstring(left_right_combined.tobytes(), left_right_combined.size, "RGBA")
            
            screen.blit(front_rear_surface, (0, 0))
            screen.blit(left_right_surface, (0, 400))

            ego_inverse_matrix = np.linalg.inv(np.array(ego_vehicle.get_transform().get_matrix()))
            ego_location = ego_vehicle.get_location()
            ego_velocity = ego_vehicle.get_velocity()
            traffic = world.get_actors().filter('vehicle.*')
            
            traffic_list = []
            for vehicle in traffic:
                if vehicle.id != ego_vehicle.id:
                    vehicle_location = vehicle.get_location()
                    vehicle_velocity = vehicle.get_velocity()

                    distance = ego_location.distance(vehicle_location)
                    if distance <= wx_max:
                        bbox_world_coords = get_bounding_box_world_coords(vehicle)
                        bbox_ego_coords = transform_to_ego_coords(bbox_world_coords, ego_inverse_matrix)
                        vehicle_bev_coord = []
                        if vehicle.type_id == 'vehicle.carlamotors.firetruck' or vehicle.type_id == 'vehicle.ford.ambulance':
                            vehicle_bev_coord.append(1) # emergency car : class 1
                        else:
                            vehicle_bev_coord.append(0) # other car : class 0
                        for coord in bbox_ego_coords:
                            bev_x, bev_y = ego_to_bev_coords(coord[0], coord[1], wx_max, wy_min, wx_interval, wy_interval)
                            if 0 <= bev_x < screen_width and 0 <= bev_y < screen_height:  # BEV 이미지 내에 있는지 확인
                                vehicle_bev_coord.append(round(bev_x/screen_width,3))
                                vehicle_bev_coord.append(round(bev_y/screen_height,3))
                                # pygame.draw.circle(screen, (255, 0, 0), (bev_x, bev_y), 5)  # 빨간색으로 점 표시
                        if len(vehicle_bev_coord) == 9:
                            traffic_list.append(vehicle_bev_coord)
            if cnt_img <= num_img:
                if cnt_scene >= 1: # 처음에 차 떨어지는 장면 저장 안하기 위해
                    save(screen, traffic_list, cnt_scene, cnt_img)
                cnt_img += 1
            elif num_img < cnt_img <= num_img*3: # Scene 간의 간격을 두기 위해
                cnt_img += 1
            elif cnt_img > num_img*3:
                cnt_img = 1
                cnt_scene += 1

            pygame.display.flip()

settings.synchronous_mode = False
world.apply_settings(settings)
front_camera.stop()
rear_camera.stop()
vehicle.destroy()
pygame.quit()
