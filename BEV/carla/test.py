import carla
import random
import time

def main():
    # CARLA 서버에 연결
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # 월드 객체 가져오기
    world = client.get_world()

    # 블루프린트 라이브러리에서 랜덤 차량 블루프린트 선택
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

    # 차량을 소환할 위치 설정
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()

    # 차량 스폰
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle is not None:
        print(f"Spawned vehicle with ID: {vehicle.type_id}")
    else:
        print("Vehicle could not be spawned.")

    # 몇 초 동안 차량을 유지한 뒤 종료
    time.sleep(5)

    # 차량 제거
    if vehicle is not None:
        vehicle.destroy()
        print("Vehicle destroyed.")

if __name__ == '__main__':
    main()
