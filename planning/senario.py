import carla, pygame, math, random, sys, os, cv2, argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from client.init import initializer

sim = initializer()
cars = 7
img_count = 0

parser = argparse.ArgumentParser()

parser.add_argument('-s', dest = 'save', action='store_true', default='save')

args = parser.parse_args()

actors = sim.world.get_actors()
vehicles = actors.filter('vehicle.*')
for vehicle in vehicles:
    vehicle.destroy()
    print(f"Removed vehicle: {vehicle.id}")

ssd_dir = '/home/jingjingtapa/다운로드/img'
scene_dir = f'{ssd_dir}/{len(os.listdir(ssd_dir))}'
for i in range(cars):
    save_dir = f'{scene_dir}/{i}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

def save_image(image, camera_index):
    global img_count
    image.convert(carla.ColorConverter.Raw)
    np_image = np.array(image.raw_data)
    np_image = np_image.reshape((image.height, image.width, 4))
    np_image = np_image[:, :, :3]  # RGB만 사용

    filename = f'{scene_dir}/{camera_index}/{img_count}.png'

    # 이미지 저장
    cv2.imwrite(filename, np_image)
    print(f"Saved image: {filename}")
    img_count += 1

def attach_rear_camera(world, vehicle, index):
    camera_transform = carla.Transform(carla.Location(x=-2, z=2.0), carla.Rotation(pitch=0, yaw=180, roll=0))
    camera = world.spawn_actor(sim.camera_bp, camera_transform, attach_to=vehicle)

    camera.listen(lambda image: save_image(image, index))

    return camera

vehicle_bp = sim.blueprint_library.filter('vehicle.*')
vehicle_bp = [bp for bp in vehicle_bp if int(bp.get_attribute('number_of_wheels')) == 4]

emergency_vehicle = sim.world.get_blueprint_library().find('vehicle.carlamotors.firetruck')



start_location = carla.Location(x=99.384415, y=0, z=0.600000)
x_distance, y_distance = 3.6, 7
cameras = []

for i in range(cars):

    vehicle = random.choice(vehicle_bp)
    if i % 2 != 0:
        spawn_location = start_location + carla.Location(x=x_distance)
    elif i % 2 == 0:
        start_location = start_location - carla.Location(y=y_distance)
        spawn_location = start_location

    spawn_point = carla.Transform(spawn_location, carla.Rotation(yaw=90))
    vehicle = sim.world.spawn_actor(vehicle, spawn_point)
    print(vehicle)
    print(spawn_point)

    if args.save == 'save':
        camera = attach_rear_camera(sim.world, vehicle, i)
        cameras.append(camera)

import time
time.sleep(5)

for camera in cameras:
    camera.stop()
    camera.destroy()