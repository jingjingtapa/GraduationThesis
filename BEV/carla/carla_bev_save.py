import numpy as np
import cv2, sys, os, random, carla, pygame
from PIL import Image
from BEVConverter import BEVConverter

# 카메라 설정
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FOV = 90

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
# world = client.load_world('Town01')
# maps = ['Town03', 'Town01', 'Town02', 'Town04', 'Town05', 'Town06', 'Town07']
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
    wheel.suspension_stiffness = 100.0
    wheel.damping_rate = 1.0
    wheel.tire_friction = 3.0

ego_vehicle.apply_physics_control(physics_control) # 변경된 물리 속성 적용
length, width = ego_vehicle.bounding_box.extent.x*2, ego_vehicle.bounding_box.extent.y*2

num_vehicles = 10
traffic_vehicles = spawn_vehicles(world, blueprint_library, num_vehicles)

wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = int(length/2), 30, 0.05, -10, 10, 0.05
bev = BEVConverter(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval)

pygame.init()
screen_width = int((wy_max - wy_min) / wy_interval)
screen_height = int(((wx_max - wx_min) / wx_interval)*2 + wx_min/wx_interval * 2)
screen = pygame.display.set_mode((screen_width, screen_height))

def attach_camera(vehicle, transform, fov=FOV):
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{CAMERA_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{CAMERA_HEIGHT}')
    camera_bp.set_attribute('fov', f'{fov}')
    camera_transform = carla.Transform(carla.Location(x=transform[0], y=transform[1], z=transform[2]),
                                       carla.Rotation(pitch=transform[3], yaw=transform[4], roll=transform[5]))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera

# 카메라 부착 (전방 카메라)
cz = 2.0
front_transform = [0.0, 0.0, cz, 0.0, 0.0, 0.0]
rear_transform = [0.0, 0.0, cz, 0.0, 180.0, 0.0]
left_transform = [0.0, 0.0, cz, 0.0, -90.0, 0.0]
right_transform = [0.0, 0.0, cz, 0.0, 90.0, 0.0]

front_camera = attach_camera(ego_vehicle, front_transform)
rear_camera = attach_camera(ego_vehicle, rear_transform)
left_camera = attach_camera(ego_vehicle, left_transform)
right_camera = attach_camera(ego_vehicle, right_transform)

K, R = bev.intrinsic(CAMERA_WIDTH, CAMERA_HEIGHT, FOV), bev.extrinsic(front_transform)

map_x_front, map_y_front, bev_height, bev_width = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval,R, K)
map_x_rear, map_y_rear, _, _ = bev.generate_direct_backward_mapping(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval, R, K)
map_x_left, map_y_left, bev_left_height, bev_left_width = bev.generate_direct_backward_mapping(int(width/2), wy_max, wy_interval, wy_min, wy_max, wx_interval, R, K)
map_x_right, map_y_right, _, _ = bev.generate_direct_backward_mapping(int(width/2), wy_max, wy_interval, wy_min, wy_max, wx_interval, R, K)

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

def get_w(vehicle):
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

def w2v(vertices_world, ego_inverse_matrix):
    vertices_ego = []
    for vertex in vertices_world:
        vertex_homogeneous = np.append(vertex, 1)  # 동차 좌표계 변환
        ego_coords = ego_inverse_matrix @ vertex_homogeneous  # Ego 좌표계로 변환
        vertices_ego.append(ego_coords[:3])  # 3D 좌표로 변환
    return np.array(vertices_ego)

def v2c(x, y, wx_max, wy_min, wx_interval, wy_interval):
    bev_x = int((y - wy_min) / wy_interval)  # y -> BEV 이미지의 x축
    bev_y = int((wx_max - x) / wx_interval)  # x -> BEV 이미지의 y축
    return bev_x, bev_y

running = True
while running:
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

    traffic = world.get_actors().filter('vehicle.*')
    for vehicle in traffic:
        if vehicle.id != ego_vehicle.id:
            vehicle_location = vehicle.get_location()
            distance = ego_location.distance(vehicle_location)
            if distance <= wx_max:
                bbox_world_coords = get_w(vehicle)
                bbox_ego_coords = w2v(bbox_world_coords, ego_inverse_matrix)
                for coord in bbox_ego_coords:
                    bev_x, bev_y = v2c(coord[0], coord[1], wx_max, wy_min, wx_interval, wy_interval)
                    if 0 <= bev_x < screen_width and 0 <= bev_y < screen_height:
                        pygame.draw.circle(screen, (255, 0, 0), (bev_x, bev_y), 5)
                        
    pygame.display.flip()

front_camera.stop()
rear_camera.stop()
vehicle.destroy()
pygame.quit()
