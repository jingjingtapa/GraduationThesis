import numpy as np
import cv2, random, carla, pygame, getpass, os, time, math
from ultralytics import YOLO
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
        R[0, 0], R[0, 1], R[0,2] = cj * ck, sj * sc - cs, sj * cc + ss
        R[1, 0], R[1, 1], R[1, 2] = cj * sk, sj * ss + cc, sj * cs - sc
        R[2, 0], R[2, 1], R[2, 2]= -sj, cj * si, cj * ci

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

    def convert_black_to_transparent(self, cv_image):
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
    
    def frontrear_leftright(self, image_front, image_rear, image_left, image_right):
        front_pil = self.convert_black_to_transparent(image_front)
        rear_pil = self.convert_black_to_transparent(cv2.flip(image_rear,-1))
        left_pil = self.convert_black_to_transparent(image_left)
        right_pil = self.convert_black_to_transparent(image_right)

        vertical_padding = Image.new("RGBA", (front_pil.width, int(wx_min / wx_interval * 2)), (0, 0, 0, 0))
        front_rear_combined = Image.new("RGBA", (front_pil.width, front_pil.height + rear_pil.height + vertical_padding.height), (0, 0, 0, 0))
        front_rear_combined.paste(front_pil, (0, 0))
        front_rear_combined.paste(vertical_padding, (0, front_pil.height))
        front_rear_combined.paste(rear_pil, (0, front_pil.height + vertical_padding.height))

        horizontal_padding = Image.new("RGBA", (0, left_pil.height), (0, 0, 0, 0))
        left_right_combined = Image.new("RGBA", (left_pil.width + right_pil.width + horizontal_padding.width, left_pil.height), (0, 0, 0, 0))
        left_right_combined.paste(left_pil, (0, 0))
        left_right_combined.paste(horizontal_padding, (left_pil.width, 0))
        left_right_combined.paste(right_pil, (left_pil.width + horizontal_padding.width, 0))

        return front_rear_combined, left_right_combined

    def screen_to_pil_image(self, screen):
        screen_array = pygame.surfarray.array3d(screen)  # Pygame Surface -> numpy 배열 (RGB 형식)
        screen_array = np.transpose(screen_array, (1, 0, 2))  # 배열 형식을 (width, height, channels)로 변환
        pil_image = Image.fromarray(screen_array)
        return pil_image
    
class CarlaInitialize:
    def __init__(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        self.world = client.get_world()

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = sampling_time
        self.world.apply_settings(self.settings)

        actors = self.world.get_actors().filter('vehicle.*')
        for actor in actors:
            actor.destroy()

        self.blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = self.blueprint_library.filter('vehicle.*')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.ego_vehicle.set_autopilot(True)

        physics_control = self.ego_vehicle.get_physics_control()
        physics_control.center_of_mass.z = -0.5 # 무게 중심 낮추기
        for wheel in physics_control.wheels: 
            wheel.suspension_stiffness = 10000.0 # 서스펜션 강도 
            wheel.damping_rate = 0.01 # 서스펜션 댐핑
            wheel.tire_friction = 3.0 # 타이어 마찰 계수 조정

        self.ego_vehicle.apply_physics_control(physics_control) # 변경된 물리 속성 적용
        self.length, self.width = self.ego_vehicle.bounding_box.extent.x*2, self.ego_vehicle.bounding_box.extent.y*2

    def spawn_traffic(self, num_vehicles):
        spawn_points = self.world.get_map().get_spawn_points()

         # 오토바이와 자전거에 포함된 태그를 정의합니다.
        exclude_tags = ['bike', 'yzf', 'ninja', 'low_rider', 'century', 'omafiets', 'crossbike', 'zx125']
        
        # 응급 차량(소방차, 구급차)을 먼저 스폰
        emergency_vehicles = ['vehicle.*firetruck', 'vehicle.*ambulance']
        for emergency in emergency_vehicles:
            vehicle_bp = random.choice(self.blueprint_library.filter(emergency))
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)

        # 일반 차량에서 오토바이와 자전거를 제외하고 스폰
        filtered_vehicles = [
            vehicle for vehicle in self.blueprint_library.filter('vehicle.*')
            if not any(tag in vehicle.tags for tag in exclude_tags)
        ]

        for i in range(num_vehicles - len(emergency_vehicles)):
            vehicle_bp = random.choice(filtered_vehicles)
            spawn_point = random.choice(spawn_points)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True)


    def attach_camera(self, vehicle, transform, sampling_time):
        camera_bp = sim.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{bev.CAMERA_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{bev.CAMERA_HEIGHT}')
        # camera_bp.set_attribute('sensor_tick', f'{sampling_time}')
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=transform[0], y=transform[1], z=transform[2]),
                                        carla.Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5]))
        camera = sim.world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        return camera

    def attatch_four_camera(self, c_z, c_pitch):
        self.front_transform = [0.0, 0.0, c_z, c_pitch, 0.0, 0.0]
        self.rear_transform = [0.0, 0.0, c_z, c_pitch, 180.0, 0.0]
        self.left_transform = [0.0, 0.0, c_z, c_pitch, -90.0, 0.0]
        self.right_transform = [0.0, 0.0, c_z, c_pitch, 90.0, 0.0]

        front_cam = self.attach_camera(self.ego_vehicle, self.front_transform, sampling_time)
        rear_cam = self.attach_camera(self.ego_vehicle, self.rear_transform, sampling_time)
        left_cam = self.attach_camera(self.ego_vehicle,self.left_transform, sampling_time)
        right_cam = self.attach_camera(self.ego_vehicle, self.right_transform, sampling_time)
        return front_cam, rear_cam, left_cam, right_cam

sim = CarlaInitialize()
sim.spawn_traffic(num_vehicles)

wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = int(sim.length/2), 15, 0.05, -10, 10, 0.05
bev = BEVConverter(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval)

pygame.init()
screen_width = int((wy_max - wy_min) / wy_interval) * 2
screen_height = int(((wx_max - wx_min) / wx_interval)*2 + wx_min/wx_interval * 2)
screen = pygame.display.set_mode((screen_width, screen_height))


c_z, c_pitch = 2.5, -30.0
front_cam, rear_cam, left_cam, right_cam = sim.attatch_four_camera(c_z, c_pitch)
K, R = bev.intrinsic(bev.CAMERA_WIDTH, bev.CAMERA_HEIGHT, bev.FOV), bev.extrinsic(sim.front_transform)

map_x_front, map_y_front, bev_height, bev_width = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval,R, K)
map_x_rear, map_y_rear, _, _ = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval, R, K)
map_x_left, map_y_left, bev_left_height, bev_left_width = bev.generate_direct_backward_mapping(math.trunc(sim.width/2), wy_max, wy_interval, -10, 10, wx_interval, R, K)
map_x_right, map_y_right, _, _ = bev.generate_direct_backward_mapping(math.trunc(sim.width/2), wy_max, wy_interval, -10, 10, wx_interval, R, K)

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

front_cam.listen(lambda image: process_image(image, 'front'))
rear_cam.listen(lambda image: process_image(image, 'rear'))
left_cam.listen(lambda image: process_image(image, 'left'))
right_cam.listen(lambda image: process_image(image, 'right'))

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

model = YOLO('/home/gunny/다운로드/GraduationThesis/BEV/yolo/runs/obb/train5/weights/best.pt')
running = True

while running:
    sim.world.tick()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((255, 255, 255))
    
    # Combine front_rear and left_right
    front_rear_combined, left_right_combined = bev.frontrear_leftright(image_front, image_rear, image_left, image_right)
    front_rear_surface = pygame.image.fromstring(front_rear_combined.tobytes(), front_rear_combined.size, "RGBA")
    left_right_surface = pygame.image.fromstring(left_right_combined.tobytes(), left_right_combined.size, "RGBA")
    screen.blit(front_rear_surface, (0, 0))
    screen.blit(left_right_surface, (0, int((wx_max-wy_max)/wx_interval)))
    
    # inference
    model_input = bev.screen_to_pil_image(screen)
    result = model.predict(model_input, nms=True, iou=0.5)
    # result = model.track(model_input, show = True, conf  = 0.4, persist = True)

    for object in result:
        if object.obb:
            for i, car in enumerate(object.obb.xyxyxyxy):
                class_label = object.obb.cls[i].item()
                # emergency car on same direction: Blue, emergency car on opposite direction: green, other car: Yellow
                color = (0, 0, 255) if class_label == 1 else (0, 255, 0) if class_label == 2 else (255, 255, 0) 
                
                # 각 꼭짓점 좌표를 (x, y) 형식으로 저장
                points = [(int(x), int(y)) for x, y in car]
                
                # 네 꼭짓점을 연결하여 사각형 그리기
                pygame.draw.polygon(screen, color, points, 2)  # '2'는 선의 두께
    


    
    ego_inverse_matrix = np.linalg.inv(np.array(sim.ego_vehicle.get_transform().get_matrix()))
    ego_location = sim.ego_vehicle.get_location()

    traffic = sim.world.get_actors().filter('vehicle.*')    
    for vehicle in traffic:
        if vehicle.id != sim.ego_vehicle.id:
            vehicle_location = vehicle.get_location()
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
                        pygame.draw.circle(screen, (255, 0, 0), (bev_x, bev_y), 3)  # 빨간색으로 점 표시
        
    pygame.display.flip()

sim.settings.synchronous_mode = False
sim.world.apply_settings(sim.settings)
front_cam.stop()
rear_cam.stop()
left_cam.stop()
right_cam.stop()
bev.ego_vehicle.destroy()
pygame.quit()