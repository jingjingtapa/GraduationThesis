import carla

def check_for_shoulder_lanes(map):
    distance_step = 5.0  # Waypoint 간의 거리 간격
    waypoints = map.generate_waypoints(distance_step)
    
    for waypoint in waypoints:
        if waypoint.lane_type == carla.LaneType.Shoulder:
            return True  # Shoulder 타입이 발견되면 True 반환
    return False  # 없으면 False 반환

# CARLA 클라이언트 및 맵 초기화
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# 모든 맵을 확인
available_maps = client.get_available_maps()
shoulder_maps = []

for map_name in available_maps:
    client.load_world(map_name)
    world = client.get_world()
    if check_for_shoulder_lanes(world.get_map()):
        shoulder_maps.append(map_name)

print("Maps with Shoulder LaneType:", shoulder_maps)
