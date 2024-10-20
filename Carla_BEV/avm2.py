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

def pad_image(image, target_width):
    """
    이미지의 너비를 target_width로 맞추기 위해 양쪽에 패딩을 추가하는 함수.
    """
    height, width = image.shape[:2]
    if width < target_width:
        pad_size = target_width - width
        left_pad = pad_size // 2
        right_pad = pad_size - left_pad
        image = cv2.copyMakeBorder(image, 0, 0, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return image

def draw_ego_vehicle(image, ego_size=50):
    """
    중앙에 사각형을 그려 ego 차량을 시각화.
    """
    height, width = image.shape[:2]
    top_left = (width // 2 - ego_size // 2, height // 2 - ego_size // 2)
    bottom_right = (width // 2 + ego_size // 2, height // 2 + ego_size // 2)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    return image

def tutorial(args):
    """
    전면, 후면, 좌, 우 카메라 데이터를 동기적으로 받아와 BEV 변환 후 실시간으로 Pygame을 통해 시각화합니다.
    """
    # Pygame 초기화
    pygame.init()
    display = pygame.display.set_mode((args.width * 2, args.height * 2), pygame.HWSURFACE | pygame.DOUBLEBUF)

    # 서버에 연결
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 빠른 갱신을 위해 설정
    world.apply_settings(settings)

    vehicle = None
    front_camera = None
    rear_camera = None
    left_camera = None
    right_camera = None

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
        vehicle.set_autopilot(True)

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

        # 좌측 카메라 설정
        left_camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(y=-1.6, z=1.6), carla.Rotation(yaw=-90)),
            attach_to=vehicle)

        # 우측 카메라 설정
        right_camera = world.spawn_actor(
            blueprint=camera_bp,
            transform=carla.Transform(carla.Location(y=1.6, z=1.6), carla.Rotation(yaw=90)),
            attach_to=vehicle)

        # 큐 생성
        front_image_queue = Queue()
        rear_image_queue = Queue()
        left_image_queue = Queue()
        right_image_queue = Queue()

        # 카메라 콜백 설정
        front_camera.listen(lambda data: sensor_callback(data, front_image_queue))
        rear_camera.listen(lambda data: sensor_callback(data, rear_image_queue))
        left_camera.listen(lambda data: sensor_callback(data, left_image_queue))
        right_camera.listen(lambda data: sensor_callback(data, right_image_queue))

        # BEV 변환을 위한 소스 및 목적지 좌표 정의
        src_points = np.float32([[150, 300], [550, 300], [50, 400], [650, 400]])
        dst_points = np.float32([[200, 0], [520, 0], [200, 720], [520, 720]])

        # 실시간 시각화 루프
        running = True
        while running:
            world.tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            try:
                front_image_data = front_image_queue.get(True, 1.0)
                rear_image_data = rear_image_queue.get(True, 1.0)
                left_image_data = left_image_queue.get(True, 1.0)
                right_image_data = right_image_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue

            # 카메라 데이터를 numpy 배열로 변환
            front_im_array = np.copy(np.frombuffer(front_image_data.raw_data, dtype=np.dtype("uint8")))
            front_im_array = np.reshape(front_im_array, (front_image_data.height, front_image_data.width, 4))[:, :, :3][:, :, ::-1]

            rear_im_array = np.copy(np.frombuffer(rear_image_data.raw_data, dtype=np.dtype("uint8")))
            rear_im_array = np.reshape(rear_im_array, (rear_image_data.height, rear_image_data.width, 4))[:, :, :3][:, :, ::-1]

            left_im_array = np.copy(np.frombuffer(left_image_data.raw_data, dtype=np.dtype("uint8")))
            left_im_array = np.reshape(left_im_array, (left_image_data.height, left_image_data.width, 4))[:, :, :3][:, :, ::-1]

            right_im_array = np.copy(np.frombuffer(right_image_data.raw_data, dtype=np.dtype("uint8")))
            right_im_array = np.reshape(right_im_array, (right_image_data.height, right_image_data.width, 4))[:, :, :3][:, :, ::-1]

            # 각 카메라의 BEV 변환
            bev_front_image = bev_transform(front_im_array, src_points, dst_points)
            bev_rear_image = bev_transform(rear_im_array, src_points, dst_points)
            bev_left_image = bev_transform(left_im_array, src_points, dst_points)
            bev_right_image = bev_transform(right_im_array, src_points, dst_points)

            # 후면 이미지를 180도 회전
            bev_rear_image = cv2.rotate(bev_rear_image, cv2.ROTATE_180)

            # 좌우 이미지를 좌우로 이동 (필요 시 조정)
            bev_left_image_shifted = shift_image(bev_left_image, 27)
            bev_right_image_shifted = shift_image(bev_right_image, -27)
            bev_front_image_padded = pad_image(bev_front_image, target_width=bev_left_image_shifted.shape[1])
            bev_rear_image_padded = pad_image(bev_rear_image, target_width=bev_left_image_shifted.shape[1])

            # 결합된 이미지 생성 (전면, 후면, 좌, 우 이미지 결합)
            top_combined = np.hstack((bev_left_image_shifted, bev_front_image_padded, bev_right_image_shifted))
            bottom_combined_padded = pad_image(bev_rear_image_padded, target_width=top_combined.shape[1])

            combined_image = np.vstack((top_combined, bottom_combined_padded))

            # ego 차량을 검은 사각형으로 시각화
            combined_image = draw_ego_vehicle(combined_image)

            # Pygame에서 결합된 이미지 시각화
            surface = pygame.surfarray.make_surface(combined_image.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            pygame.display.flip()

    finally:
        # 원래 설정 복구 및 리소스 정리
        world.apply_settings(original_settings)
        if front_camera:
            front_camera.destroy()
        if rear_camera:
            rear_camera.destroy()
        if left_camera:
            left_camera.destroy()
        if right_camera:
            right_camera.destroy()
        if vehicle:
            vehicle.destroy()
        pygame.quit()

def main():
    """메인 함수"""
    argparser = argparse.ArgumentParser(
        description='CARLA Around View Monitor (AVM) 실시간 시각화')
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

