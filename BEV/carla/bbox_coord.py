import carla
import random
import queue
import numpy as np
import cv2
from pascal_voc_writer import Writer

client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# Spawn EGO vehicle
vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

# Spawn camera
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "600")
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
vehicle.set_autopilot(True)

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)

# Get the camera projection matrix
def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# Function to convert world location to image point
def get_image_point(loc, K, world_2_camera):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(world_2_camera, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

# Camera parameters
image_w = int(camera_bp.get_attribute("image_size_x").as_int())
image_h = int(camera_bp.get_attribute("image_size_y").as_int())
fov = float(camera_bp.get_attribute("fov").as_float())

# Calculate the camera projection matrix
K = build_projection_matrix(image_w, image_h, fov)

while True:
    # Retrieve the image
    world.tick()
    image = image_queue.get()

    # Get the camera matrix 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Define file paths based on the frame ID
    frame_path = 'output/%06d' % image.frame
    image_path = frame_path + '.png'
    xml_path = frame_path + '.xml'

    # Save the image
    image.save_to_disk(image_path)

    # Initialize the Pascal VOC XML writer with the image path and size
    writer = Writer(image_path, image_w, image_h)

    # Iterate over NPC vehicles and calculate bounding boxes
    for npc in world.get_actors().filter('*vehicle*'):
        if npc.id != vehicle.id:
            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(vehicle.get_transform().location)

            # Only consider vehicles within 50m of the EGO vehicle
            if dist < 50:
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location

                # Check if the NPC is in front of the EGO vehicle
                if forward_vec.dot(ray) > 0:
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    
                    x_max, y_max = -np.inf, -np.inf
                    x_min, y_min = np.inf, np.inf

                    # Project each vertex of the bounding box to image coordinates
                    for vert in verts:
                        p = get_image_point(vert, K, world_2_camera)
                        x_max = max(x_max, p[0])
                        x_min = min(x_min, p[0])
                        y_max = max(y_max, p[1])
                        y_min = min(y_min, p[1])

                    # Add the object to the Pascal VOC annotation if within image bounds
                    if x_min > 0 and x_max < image_w and y_min > 0 and y_max < image_h:
                        writer.addObject('vehicle', int(x_min), int(y_min), int(x_max), int(y_max))

    # Save the XML file with bounding box data
    writer.save(xml_path)

    # Display the image with bounding boxes (optional visualization step)
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    cv2.imshow('Bounding Box Visualization', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
camera.stop()
camera.destroy()
vehicle.destroy()



