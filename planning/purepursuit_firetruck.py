import carla
import time
import math
import numpy as np

# Pure Pursuit Helper Function
def pure_pursuit_control(vehicle, waypoints, lookahead_distance):
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

def main():
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the world and blueprint library
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Spawn firetruck at the specified spawn point
    firetruck_bp = blueprint_library.filter('firetruck')[0]
    spawn_point = carla.Transform(carla.Location(x=0, y=-60.899998, z=0.600000))
    firetruck = world.spawn_actor(firetruck_bp, spawn_point)

    # Set the spectator to the firetruck
    spectator = world.get_spectator()
    spectator.set_transform(firetruck.get_transform())

    # Get the map and current waypoint at spawn point
    map = world.get_map()
    current_waypoint = map.get_waypoint(spawn_point.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    
    # Generate waypoints to follow the current lane
    waypoints = []
    next_waypoint = current_waypoint
    while next_waypoint and len(waypoints) < 100:  # Limit the number of waypoints
        waypoints.append(next_waypoint)
        next_waypoints = next_waypoint.next(2.0)  # Get the next waypoint 2m ahead
        if len(next_waypoints) == 0:
            break
        next_waypoint = next_waypoints[0]  # Move to the next waypoint

    # Run the simulation for 20 seconds
    start_time = time.time()
    lookahead_distance = 0.5  # Lookahead distance for Pure Pursuit

    while time.time() - start_time < 25:
        # Calculate control using Pure Pursuit
        steering_angle, throttle = pure_pursuit_control(firetruck, waypoints, lookahead_distance)

        # Apply control to the firetruck
        control = carla.VehicleControl()
        control.steer = np.clip(steering_angle, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        firetruck.apply_control(control)

        # Sleep for a short duration to maintain the loop at a reasonable frequency
        time.sleep(0.05)

    # Destroy the firetruck actor at the end
    firetruck.destroy()

if __name__ == '__main__':
    main()

