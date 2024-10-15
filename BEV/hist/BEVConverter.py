from cityscapesscripts.helpers.box3dImageTransform import Camera,Box3dImageTransform, CRS_V, CRS_C, CRS_S
from cityscapesscripts.helpers.annotation import CsBbox3d
import cv2, json, os, math
import numpy as np

class BEVConverter:
    def __init__(self, wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval):
        self.wx_min, self.wx_max, self.wx_interval, self.wy_min, self.wy_max, self.wy_interval = wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval

    def read_json(self,dir):
        with open(dir, 'r') as file:
            data = json.load(file)
        return data

    def rotation_from_euler(self, roll=1., pitch=1., yaw=1.):
        # roll, pitch, yaw -> R [4, 4]

        si, sj, sk = np.sin(roll), np.sin(pitch), np.sin(yaw)
        ci, cj, ck = np.cos(roll), np.cos(pitch), np.cos(yaw)
        cc, cs = ci * ck, ci * sk
        sc, ss = si * ck, si * sk

        R = np.identity(4)
        R[0, 0] = cj * ck
        R[0, 1] = sj * sc - cs
        R[0, 2] = sj * cc + ss
        R[1, 0] = cj * sk
        R[1, 1] = sj * ss + cc
        R[1, 2] = sj * cs - sc
        R[2, 0] = -sj
        R[2, 1] = cj * si
        R[2, 2] = cj * ci
        return R

    def translation_matrix(self, vector):
        # Translation vector -> T [4, 4]
        M = np.identity(4)
        M[:3, 3] = vector[:3]
        return M

    def motion_cancel(self, cal):
        imu = np.array(cal['sensor']['sensor_T_ISO_8855']) # 3*4
        rotation, translation = imu[:2,:2].T, -imu[:2,3]
        
        motion_cancel_mat = np.identity(3)
        motion_cancel_mat[:2,:2] = rotation
        motion_cancel_mat[:2,2] = translation

        return motion_cancel_mat

    def load_camera_params(self, cal): #c: calibration
        fx, fy = cal['intrinsic']['fx'], cal['intrinsic']['fy']
        u0, v0 = cal['intrinsic']['u0'], cal['intrinsic']['v0']

        pitch, roll, yaw = cal['extrinsic']['pitch'], cal['extrinsic']['roll'], cal['extrinsic']['yaw']
        x, y, z = cal['extrinsic']['x'], cal['extrinsic']['y'], cal['extrinsic']['z']

        baseline = cal['extrinsic']['baseline']
        # Intrinsic
        K = np.array([[fx, 0, u0, 0],
                    [0, fy, v0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

        # Extrinsic
        R_veh2cam = np.transpose(self.rotation_from_euler(roll, pitch, yaw))
        T_veh2cam = self.translation_matrix((-x-baseline/2, -y, -z))

        # Rotate to camera coordinates
        R = np.array([[0., -1., 0., 0.],
                    [0., 0., -1., 0.],
                    [1., 0., 0., 0.],
                    [0., 0., 0., 1.]])

        RT = R @ R_veh2cam @ T_veh2cam
        return RT, K

    def generate_direct_backward_mapping(self,
        world_x_min, world_x_max, world_x_interval, 
        world_y_min, world_y_max, world_y_interval, 
        extrinsic, intrinsic, motion_cancel_mat):
        
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

                # uv_coord = motion_cancel_mat @ uv_coord

                map_x[i][j] = uv_coord[0]
                map_y[i][j] = uv_coord[1]
                
        return map_x, map_y, output_height, output_width
        
    def get_BEV_image(self, image_dir, cal_dir, bbox_dir):
        image, calibration, box = cv2.imread(image_dir), self.read_json(cal_dir), self.read_json(bbox_dir)
        extrinsic, intrinsic = self.load_camera_params(calibration)
        motion_cancel_mat = self.motion_cancel(box)
        map_x, map_y, bev_height, bev_width = self.generate_direct_backward_mapping(self.wx_min, self.wx_max, self.wx_interval, self.wy_min, self.wy_max, self.wy_interval, extrinsic, intrinsic, motion_cancel_mat)
        bev_image = cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        
        camera = Camera(fx=box["sensor"]["fx"],fy=box["sensor"]["fy"],
                u0=box["sensor"]["u0"],v0=box["sensor"]["v0"],
                sensor_T_ISO_8855=box["sensor"]["sensor_T_ISO_8855"])
        box3d_annotation = Box3dImageTransform(camera=camera)

        obj = CsBbox3d()

        bev_objects = []
        for object in box['objects']:
            obj.fromJsonText(object)
            box3d_annotation.initialize_box_from_annotation(obj, coordinate_system=CRS_V)
            box_vertices_C = box3d_annotation.get_vertices(coordinate_system=CRS_C)
            bottom_vertice = dict()
            for loc, pv_coord in box_vertices_C.items():
                if loc in ['FRB', 'FLB', 'BLB', 'BRB']:
                    bottom_vertice[loc] = [pv_coord[0],pv_coord[1]]
                # class, x_center, y_center, width, height
                if object['label'] == 'car': cls = 1
                else: cls = 0
            
            width = int((math.sqrt((bottom_vertice['FLB'][0]-bottom_vertice['FRB'][0])**2 + (bottom_vertice['FLB'][1]-bottom_vertice['FRB'][1])**2))/self.wy_interval)
            height = int((math.sqrt((bottom_vertice['FRB'][0]-bottom_vertice['BRB'][0])**2 + (bottom_vertice['FRB'][1]-bottom_vertice['BRB'][1])**2))/self.wx_interval)
            center_vx, center_vy = int((bottom_vertice['FRB'][0]+bottom_vertice['BLB'][0])/2), int((bottom_vertice['FRB'][1]+bottom_vertice['BLB'][1])/2)
            center_ux, center_uy = int(0.5*bev_width - center_vy/self.wy_interval), int(bev_height - (center_vx - self.wx_min)/self.wx_interval)
            
            if center_ux >= 0 and center_uy >= 0:                              
                # bev_objects.append([cls, center_ux, center_uy, width, height])
                bev_objects.append([cls, bottom_vertice['FLB'][0], bottom_vertice['FLB'][1], bottom_vertice['BRB'][0], bottom_vertice['BRB'][1]])

        return bev_image, bev_objects
    