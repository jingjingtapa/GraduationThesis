import glob
import os
import sys
import carla
import argparse
from queue import Queue, Empty
import numpy as np
import pygame
import cv2

def sensor_callback(data, queue):
    """
    카메라 데이터를 큐에 저장하는 콜백 함수.
    """
    queue.put(data)

def bev_transform(image, src_points, dst_points):
    """
    BEV 변환을 수행하는 함수.
    """
    H, _ = cv2.findHomography(src_points, dst_points)
    bev_image = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))
    return bev_image

def shift_image(image, x_shift):
    """
    이미지를 x축 방향으로 x_shift만큼 이동시키는 함수.
    """
    height, width = image.shape[:2]
    translation_matrix = np.float32([[1, 0, x_shift], [0, 1, 0]])
    shifted_image = cv2.warpAffine(image, translation_matrix, (width, height))
    return shifted_image

def process_keyboard(vehicle):
    """
    키보드를 통한 수동 제어.
    """
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP]:
        vehicle.apply_control(carla.VehicleControl(throttle=0.5))
    elif keys[pygame.K_DOWN]:
        vehicle.apply_control(carla.VehicleControl(brake=0.5))
    elif keys[pygame.K_LEFT]:
        vehicle.apply_control(carla.VehicleControl(steer=-0.5))
    elif keys[pygame.K_RIGHT]:
        vehicle.apply_control(carla.VehicleControl(steer=0.5))
    else:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0))

def tutorial(args):
    """
    이 함수는 전면 및 후면 카메라 데이터를 받아와 BEV 변환 후 실시간으로 Pygame을 통해 시각화하고,
    수동 제어가 가능하도록 수정된 코드입니다.
    """
    # Pygame 초기화
    pygame.init()
    display = pygame.display.set_mode((args.width, args.height * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    # 서버에 연결
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 빠른 갱신을 위해 설정
    world.apply_settings(settings)

    vehicle = None
    front_camera = None
    rear_camera = None
    traffic_vehicles = []

    try:
        # 차량 및 카메라 블루프린트 설정
        vehicle_bp = bp_lib.filter("vehicle.lincoln.mkz_2017")[0]
        camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
        camera_bp.set_attribute("image_size_x", str(args.width))
        camera_bp.set_attribute("image_size_y", str(args.height))

        # 차량 스폰
        vehicle = world.spawn_actor(
            blueprint=vehicle_bp,
            transform=world.get_map().get_spawn_points()[0])

        # 전면 카메라 설정
        front_camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=1.6, z=1.6)),
            attach_to=vehicle)

        # 후면 카메라 설정
        rear_camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(x=-1.6, z=1.6), carla.Rotation(yaw=180)),
            attach_to=vehicle)

        # 큐 생성
        front_image_queue = Queue()
        rear_image_queue = Queue()

        # 카메라 콜백 설정
        front_camera.listen(lambda data: sensor_callback(data, front_image_queue))
        rear_camera.listen(lambda data: sensor_callback(data, rear_image_queue))

        # BEV 변환을 위한 소스 및 목적지 좌표 정의
        src_points = np.float32([[100, 250], [580, 250], [50, 400], [650, 400]])

	# dst_points: 변환된 이미지에서 나타낼 좌표 (더 넓은 영역에 해당하는 변환 좌표 설정)
        dst_points = np.float32([[150, 0], [600, 0], [150, 720], [600, 720]])
        traffic_blueprints = bp_lib.filter("vehicle.*")  # 모든 차량 블루프린트를 가져옴
        spawn_points = world.get_map().get_spawn_points()
        num_traffic_vehicles = 30  # 스폰할 교통 차량 수

        for _ in range(num_traffic_vehicles):
            traffic_vehicle_bp = np.random.choice(traffic_blueprints)
            spawn_point = np.random.choice(spawn_points)
            traffic_vehicle = world.try_spawn_actor(traffic_vehicle_bp, spawn_point)
            if traffic_vehicle:
                traffic_vehicle.set_autopilot(True)  # 자율 주행 모드 설정
                traffic_vehicles.append(traffic_vehicle)

        # 실시간 시각화 루프
        running = True
        while running:
            world.tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 차량을 키보드로 수동 제어
            process_keyboard(vehicle)

            try:
                front_image_data = front_image_queue.get(True, 1.0)
                rear_image_data = rear_image_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            # 전면 카메라 데이터를 numpy 배열로 변환
            front_im_array = np.copy(np.frombuffer(front_image_data.raw_data, dtype=np.dtype("uint8")))
            front_im_array = np.reshape(front_im_array, (front_image_data.height, front_image_data.width, 4))
            front_im_array = front_im_array[:, :, :3][:, :, ::-1]  # BGRA to RGB

            # 후면 카메라 데이터를 numpy 배열로 변환
            rear_im_array = np.copy(np.frombuffer(rear_image_data.raw_data, dtype=np.dtype("uint8")))
            rear_im_array = np.reshape(rear_im_array, (rear_image_data.height, rear_image_data.width, 4))
            rear_im_array = rear_im_array[:, :, :3][:, :, ::-1]  # BGRA to RGB

            # 전면 및 후면 이미지의 BEV 변환
            bev_front_image = bev_transform(front_im_array, src_points, dst_points)
            bev_rear_image = bev_transform(rear_im_array, src_points, dst_points)

            # 후면 이미지를 180도 회전
            bev_rear_image = cv2.rotate(bev_rear_image, cv2.ROTATE_180)

            # 후면 이미지를 좌우로 x_shift 만큼 이동
            x_shift = 27
            bev_rear_image_shifted = shift_image(bev_rear_image, x_shift)

            # 결합된 이미지 생성 (전면 이미지 위, 후면 이미지 아래)
            combined_image = np.vstack((bev_front_image, bev_rear_image_shifted))

            # Pygame에서 결합된 이미지 시각화
            surface = pygame.surfarray.make_surface(combined_image.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()

            # FPS를 60으로 유지
            clock.tick(60)

    finally:
        # 원래 설정 복구 및 리소스 정리
        world.apply_settings(original_settings)
        if front_camera:
            front_camera.destroy()
        if rear_camera:
            rear_camera.destroy()
        if vehicle:
            vehicle.destroy()
        for traffic_vehicle in traffic_vehicles:
            traffic_vehicle.destroy()
        pygame.quit()

def main():
    """메인 함수"""
    argparser = argparse.ArgumentParser(
        description='CARLA Front and Rear Camera BEV 실시간 시각화 및 수동 제어')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='호스트 서버 IP (기본값: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP 포트 (기본값: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='680x420',
        help='화면 해상도 (기본값: 1280x720)')
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        tutorial(args)

    except KeyboardInterrupt:
        print('\n사용자에 의해 취소됨. 종료합니다.')


if __name__ == '__main__':
    main()

