import carla

class initializer:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        # self.world = self.client.load_world('Town01_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Buildings)
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
    