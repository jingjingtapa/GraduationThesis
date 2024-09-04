import carla, math

class initializer:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        # self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.traffic_manager = self.client.get_trafficmanager(8000)

        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '800')
        self.camera_bp.set_attribute('image_size_y', '600')
        self.camera_bp.set_attribute('fov', '90')

        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        self.DISPLAY_WIDTH = 800
        self.DISPLAY_HEIGHT = 600

    def pure_pursuit_control(self,vehicle, waypoints, lookahead_distance):
        vehicle_transform = vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)

        # Find the nearest waypoint that is at least lookahead_distance away
        closest_distance = float('inf')
        closest_waypoint = None

        for waypoint in waypoints:
            distance = vehicle_location.distance(waypoint.transform.location)
            if distance > lookahead_distance and distance < closest_distance:
                closest_distance = distance
                closest_waypoint = waypoint

        if closest_waypoint is None:
            return 0.0, 0.0  # No suitable waypoint found

        # Calculate the target point relative to the vehicle's position
        dx = closest_waypoint.transform.location.x - vehicle_location.x
        dy = closest_waypoint.transform.location.y - vehicle_location.y

        # Transform to the vehicle's coordinate system
        transformed_x = dx * math.cos(-vehicle_yaw) - dy * math.sin(-vehicle_yaw)
        transformed_y = dx * math.sin(-vehicle_yaw) + dy * math.cos(-vehicle_yaw)

        # Calculate steering angle using Pure Pursuit
        steering_angle = math.atan2(2 * transformed_y, lookahead_distance)

        # Assume constant speed for simplicity (e.g., 15 m/s)
        target_speed = 7.0
        current_speed = vehicle.get_velocity().length()

        # PID controller for speed
        throttle = 0.5 * (target_speed - current_speed)

        return steering_angle, throttle