import carla
import pygame
import math
import random
import sys
import os
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from client.init import initializer

# 시뮬레이터 초기화
sim = initializer()
world = sim.client.load_world('Town06')
sim.world = world
# # 맵에 남아있던 차량 삭제
# actors = sim.world.get_actors()
# vehicles = actors.filter('vehicle.*')
# for vehicle in vehicles:
#     vehicle.destroy()
#     print(f"Removed vehicle: {vehicle.id}")

# # 블루프린트 라이브러리 가져오기
# blueprint_library = sim.world.get_blueprint_library()

# # 모든 차량 블루프린트와 firetruck 블루프린트 가져오기
# vehicle_bp = sim.blueprint_library.filter('vehicle.*')
# firetruck_blueprint = blueprint_library.filter('vehicle.*firetruck*')[0]

# # 스폰할 위치와 방향 설정
# spawn_point1 = carla.Transform(carla.Location(x=0, y=-60.899998, z=0.600000))
# spawn_point2 = carla.Transform(carla.Location(x=-8.8888, y=-58.000, z=0.600000))  # 차 크기로 조정예정

# # 랜덤 차량 블루프린트 선택
# vehicle_blueprint = random.choice(vehicle_bp)

# # 차량 스폰
# try:
#     vehicle1 = sim.world.spawn_actor(vehicle_blueprint, spawn_point1)
#     print(f"Vehicle1 spawned: {vehicle1.type_id} at {spawn_point1}")
# except Exception as e:
#     print(f"Failed to spawn vehicle1: {e}")

# try:
#     vehicle2 = sim.world.spawn_actor(firetruck_blueprint, spawn_point2)
#     print(f"Vehicle2 spawned: {vehicle2.type_id} at {spawn_point2}")
# except Exception as e:
#     print(f"Failed to spawn vehicle2: {e}")

# import threading

# def change_lane(vehicle, change_to_right=True):
#     # 현재 차량 위치에서 웨이포인트를 가져옴
#     current_waypoint = sim.world.get_map().get_waypoint(vehicle.get_location())
#     # 오른쪽 또는 왼쪽 차로로 변경
#     if change_to_right:
#         target_waypoint = current_waypoint.get_right_lane()
#     else:
#         target_waypoint = current_waypoint.get_left_lane()

#     # 목표 웨이포인트로 차량을 이동
#     if target_waypoint is not None:
#         vehicle.set_autopilot(False)
#         while True:
#             current_location = vehicle.get_location()
#             target_location = target_waypoint.transform.location
#             direction_vector = target_location - current_location
#             direction_norm = math.sqrt(direction_vector.x**2 + direction_vector.y**2 + direction_vector.z**2)
#             norm_direction_vector = carla.Vector3D(direction_vector.x / direction_norm, direction_vector.y / direction_norm, direction_vector.z / direction_norm)

#             # steer 계산
#             steer = 0.5 * norm_direction_vector.y # 간단한 조향 계산, 실제 상황에서는 더 복잡한 계산 필요

#             # 차량 제어
#             vehicle.apply_control(carla.VehicleControl(throttle=0.25, steer=steer))
#             time.sleep(0.05)  # 주기적으로 업데이트

#             # 목표 지점에 충분히 가까워지면 중지
#             if current_location.distance(target_location) < 2.0:
#                 print(f"{vehicle.type_id} reached the target lane")
#                 vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
#                 break

# # 각 차량을 별도의 스레드에서 차로 변경하도록 설정
# thread1 = threading.Thread(target=change_lane, args=(vehicle1, False))  # vehicle1 왼쪽 차로로 변경
# thread2 = threading.Thread(target=change_lane, args=(vehicle2, True))   # vehicle2 오른쪽 차로로 변경

# # 스레드 시작
# thread1.start()
# thread2.start()

# # 모든 스레드가 종료될 때까지 대기
# thread1.join()
# thread2.join()




