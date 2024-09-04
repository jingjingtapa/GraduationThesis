import carla
import math
import time

def calculate_steering_angle(vehicle_transform, lookahead_waypoint):
    """ Pure Pursuit 스티어링 각도 계산 """
    vehicle_location = vehicle_transform.location
    waypoint_location = lookahead_waypoint.transform.location

    # 방향 벡터 계산
    dx = waypoint_location.x - vehicle_location.x
    dy = waypoint_location.y - vehicle_location.y

    # 차량의 헤딩과 타겟 포인트 간의 각도 차이 계산
    heading = math.radians(vehicle_transform.rotation.yaw)
    target_angle = math.atan2(dy, dx)
    angle_diff = target_angle - heading

    # 각도 정규화
    while angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    return angle_diff

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    for vehicle in vehicles:
        vehicle.destroy()
        print(f"Removed vehicle: {vehicle.id}")

    try:
        # 차량 스폰
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.audi.a2')
        spawn_point = carla.Transform(carla.Location(x=62.3, y=-60.9, z=0), carla.Rotation(yaw=0))
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        # 웨이포인트 경로 생성
        map = world.get_map()
        start_waypoint = map.get_waypoint(carla.Location(x=62.3, y=-60.9, z=0))
        end_waypoint = map.get_waypoint(carla.Location(x=98.4, y=-32.0, z=0))

        waypoints = []
        current_waypoint = start_waypoint
        while current_waypoint and current_waypoint.transform.location.distance(end_waypoint.transform.location) > 1.0:
            waypoints.append(current_waypoint)
            next_waypoints = current_waypoint.next(3.0)
            if next_waypoints:
                current_waypoint = next_waypoints[0]
            else:
                break

        # Pure Pursuit 알고리즘 실행
        for waypoint in waypoints:
            control = carla.VehicleControl()
            vehicle_transform = vehicle.get_transform()
            steering_angle = calculate_steering_angle(vehicle_transform, waypoint)
            control.steer = max(-1.0, min(1.0, steering_angle))
            control.throttle = 0.5  # 일정 속도 유지
            vehicle.apply_control(control)
            time.sleep(0.05)

    finally:
        if vehicle is not None:
            vehicle.destroy()

if __name__ == '__main__':
    main()
