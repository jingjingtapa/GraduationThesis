import sys, os, random, time, carla
import numpy as np
import cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from client.init import initializer
img_counter, frame_counter = 0, 0
sampling_rate = 10

def process_image(image, save_dir):
    global img_counter
    global frame_counter
    if frame_counter % sampling_rate == 0:
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_data = image_data.reshape((image.height, image.width, 4))  # BGRA
        image_bgr = image_data[:, :, :3]  # BGR로 변환 (A 채널 제거)
        image_name = os.path.join(save_dir, f'image_{img_counter:04d}.png')
        cv2.imwrite(image_name, image_bgr)
        img_counter += 1
    frame_counter += 1
    
def main():
    ssd_dir = '/mnt/wdblack/imgs'
    save_dir = f'{ssd_dir}/{len(os.listdir(ssd_dir))}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        sim = initializer()
        
        spawn_point = random.choice(sim.spawn_points)
        vehicle_bp = sim.blueprint_library.filter('vehicle.tesla.model3')[0]
        vehicle = sim.world.spawn_actor(vehicle_bp, spawn_point)

        camera_transform = carla.Transform(carla.Location(x=-2.5, z=1.0), carla.Rotation(yaw=180))

        camera = sim.world.spawn_actor(sim.camera_bp, camera_transform, attach_to = vehicle)
        camera.listen(lambda image: process_image(image, save_dir=save_dir))

        traffic_manager = sim.client.get_trafficmanager(8000)
        vehicle.set_autopilot(True, traffic_manager.get_port())
        
        print("Autopilot enabled.")
        while True:
            sim.world.tick()  # 시뮬레이션 업데이트
            time.sleep(0.1)  # 약간의 대기 시간을 줘서 CPU 부하 방지

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        if camera is not None:
            camera.destroy()
            print("Camera destroyed.")
        if vehicle is not None:
            vehicle.destroy()
            print("Vehicle destroyed.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
