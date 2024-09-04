import carla, pygame, math, random, sys, os, cv2, argparse, time
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from client.init import initializer

sim = initializer()
cars = 7
img_count = 0

parser = argparse.ArgumentParser()

parser.add_argument('-s', dest = 'save', action='store', default='save')

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
    np_image = np_image[:, :, :3]  # Use only RGB

    filename = f'{scene_dir}/{camera_index}/{img_count}.png'
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

firetruck_bp = sim.blueprint_library.filter('firetruck')[0]
spawn_point = carla.Transform(carla.Location(x=62.7, y=-61.0, z=0.6), carla.Rotation(yaw=0))
firetruck = sim.world.spawn_actor(firetruck_bp, spawn_point)

map = sim.world.get_map()
current_waypoint = map.get_waypoint(spawn_point.location, project_to_road=True, lane_type=carla.LaneType.Driving)

waypoints = []
next_waypoint = current_waypoint
while next_waypoint and len(waypoints) < 100:
    waypoints.append(next_waypoint)
    next_waypoints = next_waypoint.next(0.5)
    if len(next_waypoints) == 0:
        break
    next_waypoint = next_waypoints[0]

start_time = time.time()
lookahead_distance = 0.6

while time.time() - start_time < 11:
    steering_angle, throttle = sim.pure_pursuit_control(firetruck, waypoints, lookahead_distance)
    control = carla.VehicleControl()
    control.steer = np.clip(steering_angle, -1.0, 1.0)
    control.throttle = np.clip(throttle, 0.0, 1.0)
    firetruck.apply_control(control)
    time.sleep(0.05)

firetruck.destroy()

for camera in cameras:
    camera.stop()
    camera.destroy()
