import carla
import math
import random
import time
import queue
import numpy as np
import cv2

client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# spawn vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
vehicle.set_autopilot(True)

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]

    # Get the world to camera matrix
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# Get the attributes from the camera
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)
K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)


# Retrieve all bounding boxes for traffic lights within the level
bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)

# Filter the list to extract bounding boxes within a 50m radius
nearby_bboxes = []
for bbox in bounding_box_set:
    if bbox.location.distance(vehicle.get_transform().location) < 50:
        nearby_bboxes.append(bbox)

edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

for i in range(50):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if npc:
        npc.set_autopilot(True)


# Retrieve the first image
world.tick()
image = image_queue.get()

# Reshape the raw data into an RGB array
img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 

# Display the image in an OpenCV display window
cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
cv2.imshow('ImageWindowName',img)
cv2.waitKey(1)


def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

while True:
    # Retrieve and reshape the image
    world.tick()
    image = image_queue.get()

    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    # Get the camera matrix 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())


    for npc in world.get_actors().filter('*vehicle*'):

        # Filter out the ego vehicle
        if npc.id != vehicle.id:

            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(vehicle.get_transform().location)

            # Filter for the vehicles within 50m
            if dist < 50:

            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the other vehicle. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location

                if forward_vec.dot(ray) > 0:
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    for edge in edges:
                        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                        p2 = get_image_point(verts[edge[1]],  K, world_2_camera)

                        p1_in_canvas = point_in_canvas(p1, image_h, image_w)
                        p2_in_canvas = point_in_canvas(p2, image_h, image_w)

                        if not p1_in_canvas and not p2_in_canvas:
                            continue

                        ray0 = verts[edge[0]] - camera.get_transform().location
                        ray1 = verts[edge[1]] - camera.get_transform().location
                        cam_forward_vec = camera.get_transform().get_forward_vector()

                        # One of the vertex is behind the camera
                        if not (cam_forward_vec.dot(ray0) > 0):
                            p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
                        if not (cam_forward_vec.dot(ray1) > 0):
                            p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)

                        cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)        

    cv2.imshow('ImageWindowName',img)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()

# 종료할 때 카메라 객체를 안전하게 삭제합니다.
camera.stop()
camera.destroy()
vehicle.destroy()
