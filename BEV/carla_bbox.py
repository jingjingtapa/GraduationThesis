import carla, random
import numpy as np

def get_bounding_box_world_coords(vehicle):
    """
    Get the 8 vertices of a vehicle's bounding box in world coordinates.
    """
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
        vertex_homogeneous = np.append(vertex, 1)  # 동차 좌표계 변환
        world_coords = vehicle_matrix @ vertex_homogeneous  # 월드 좌표계로 변환
        vertices_world.append(world_coords[:3])  # 3D 좌표로 변환
    return np.array(vertices_world)

def transform_to_ego_coords(vertices_world, ego_inverse_matrix):
    """
    Transform the 3D points from world coordinates to Ego vehicle's local coordinates.
    """
    vertices_ego = []
    for vertex in vertices_world:
        vertex_homogeneous = np.append(vertex, 1)  # 동차 좌표계 변환
        ego_coords = ego_inverse_matrix @ vertex_homogeneous  # Ego 좌표계로 변환
        vertices_ego.append(ego_coords[:3])  # 3D 좌표로 변환
    return np.array(vertices_ego)

def spawn_vehicles(world, blueprint_library, num_vehicles):
    """
    Spawn multiple vehicles at random spawn points.
    """
    spawn_points = world.get_map().get_spawn_points()
    vehicles_list = []
    
    for i in range(num_vehicles):
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True)  # Enable autopilot for spawned vehicles
            vehicles_list.append(vehicle)
    
    return vehicles_list

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
    
    # Ego vehicle spawn
    blueprint_library = world.get_blueprint_library()
    ego_vehicle_bp = blueprint_library.filter('vehicle.lincoln.mkz_2017')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    ego_vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)
    ego_vehicle.set_autopilot(True)

    num_vehicles = 10  # Number of additional vehicles to spawn
    traffic_vehicles = spawn_vehicles(world, blueprint_library, num_vehicles)

    try:
        while True:
            world.tick()

            # Ego 차량의 변환 행렬
            ego_transform = ego_vehicle.get_transform()
            ego_inverse_matrix = np.linalg.inv(np.array(ego_transform.get_matrix()))

            # Ego 차량의 위치
            ego_location = ego_vehicle.get_location()

            # 주변 차량의 바운딩 박스 좌표를 Ego 좌표계로 변환
            vehicles = world.get_actors().filter('vehicle.*')
            for vehicle in vehicles:
                if vehicle.id != ego_vehicle.id:
                    # 차량의 위치 가져오기
                    vehicle_location = vehicle.get_location()
                    # Ego 차량과의 거리 계산
                    distance = ego_location.distance(vehicle_location)

                    # 반경 10m 이내의 차량만 처리
                    if distance <= 10:
                        bbox_world_coords = get_bounding_box_world_coords(vehicle)
                        bbox_ego_coords = transform_to_ego_coords(bbox_world_coords, ego_inverse_matrix)
                        print(f"Vehicle {vehicle.id} (distance {distance:.2f}m) Bounding Box in Ego Coordinates: {bbox_ego_coords}")

    finally:
        print("Cleaning up vehicles...")
        ego_vehicle.destroy()
        for vehicle in traffic_vehicles:
            vehicle.destroy()
        print("Simulation ended.")

if __name__ == '__main__':
    main()
