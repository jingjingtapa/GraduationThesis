import carla, pygame, math, random, sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from client.init import initializer

sim = initializer()
vehicle_bp = sim.blueprint_library.filter('vehicle.*')
# spawn_point = random.choice(sim.spawn_points)
spawn_point = carla.Transform(carla.Location(x=0, y=-60.899998, z=0.600000))
vehicle = random.choice(vehicle_bp)
vehicle = sim.world.spawn_actor(vehicle, spawn_point) 
print(vehicle)
print(spawn_point)