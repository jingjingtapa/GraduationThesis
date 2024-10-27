from cityscapesscripts.helpers.box3dImageTransform import Camera,Box3dImageTransform, CRS_V, CRS_C, CRS_S
from cityscapesscripts.helpers.annotation import CsBbox3d
import cv2, json, os, math
import numpy as np

class BEVConverter:
    def __init__(self, wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval):
        self.wx_min, self.wx_max, self.wx_interval, self.wy_min, self.wy_max, self.wy_interval = wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval
        self.CAMERA_WIDTH, self.CAMERA_HEIGHT, self.FOV = 640, 480, 90

    def intrinsic(self, camera_width, camera_height, fov):
        f_x = camera_width / (2 * np.tan(np.radians(fov / 2)))
        f_y = f_x
        c_x, c_y = camera_width / 2, camera_height / 2

        K = np.array([[f_x, 0, c_x],
                    [0, f_y, c_y],
                    [0, 0, 1]])
        return K

    def extrinsic(self, t):
        x, y, z = t[0], t[1], t[2]
        roll, pitch, yaw = t[3], t[4], t[5]
        
        R = np.identity(4)
        R[0, 3], R[1, 3], R[2, 3] = -x, -y, -z

        r = np.array([[0., -1., 0., 0.],
                    [0., 0., -1., 0.],
                    [1., 0., 0., 0.],
                    [0., 0., 0., 1.]])
        RT = r @ R
        return RT

    def generate_direct_backward_mapping(self,
        world_x_min, world_x_max, world_x_interval, 
        world_y_min, world_y_max, world_y_interval, 
        extrinsic, intrinsic):
        
        world_x_coords = np.arange(world_x_max, world_x_min, -world_x_interval)
        world_y_coords = np.arange(world_y_max, world_y_min, -world_y_interval)
        
        output_height = len(world_x_coords)
        output_width = len(world_y_coords)
        
        map_x = np.zeros((output_height, output_width)).astype(np.float32)
        map_y = np.zeros((output_height, output_width)).astype(np.float32)
        world_z = 0
        for i, world_x in enumerate(world_x_coords):
            for j, world_y in enumerate(world_y_coords):
                # [world_x, world_y, 0, 1] -> [u, v, 1]
                world_coord = [world_x, world_y, world_z, 1]
                camera_coord = extrinsic[:3, :] @ world_coord # 3*4 * 4*1 = 3*1
                uv_coord = intrinsic[:3, :3] @ camera_coord # 3*3 * 3*1 = 3*1
                uv_coord /= uv_coord[2]

                map_x[i][j] = uv_coord[0]
                map_y[i][j] = uv_coord[1]
                
        return map_x, map_y, output_height, output_width
        
    def get_BEV_image(self, image, map_x, map_y):
        bev_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return bev_image
    
