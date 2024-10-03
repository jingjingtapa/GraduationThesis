import cv2, json, os, math
import numpy as np

def read_json(dir):
    with open(dir, 'r') as file:
        data = json.load(file)
    return data

def rotation_from_euler(roll=1., pitch=1., yaw=1.):
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

def euler_from_rotation(R):
    assert(R.shape == (3, 3))

    if R[2, 0] != 1 and R[2, 0] != -1:
        pitch1 = -np.arcsin(R[2, 0])
        pitch2 = np.pi - pitch1
        roll1 = np.arctan2(R[2, 1] / np.cos(pitch1), R[2, 2] / np.cos(pitch1))
        roll2 = np.arctan2(R[2, 1] / np.cos(pitch2), R[2, 2] / np.cos(pitch2))
        yaw1 = np.arctan2(R[1, 0] / np.cos(pitch1), R[0, 0] / np.cos(pitch1))
        yaw2 = np.arctan2(R[1, 0] / np.cos(pitch2), R[0, 0] / np.cos(pitch2))
        return (roll1, pitch1, yaw1), (roll2, pitch2, yaw2)
    else:
        yaw = 0  # can set to anything, it's the gimbal lock case
        if R[2, 0] == -1:
            pitch = np.pi / 2
            roll = yaw + np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])
        return (roll, pitch, yaw), (None, None, None)

def translation_matrix(vector):
    # Translation vector -> T [4, 4]
    M = np.identity(4)
    M[:3, 3] = vector[:3]
    return M

def motion_cancel(cal):
    imu = np.array(cal['sensor']['sensor_T_ISO_8855']) # 3*4
    rotation, translation = imu[:,:3], imu[:2,2]
    
    euler_angle_set_1, euler_angle_set_2 = euler_from_rotation(rotation)
    
    roll, pitch, yaw = euler_angle_set_1
    
    R_roll = np.array([
        [np.cos(roll), -np.sin(roll)],
        [np.sin(roll), np.cos(roll)]
    ])
    
    R_pitch = np.array([
        [np.cos(pitch), -np.sin(pitch)],
        [np.sin(pitch), np.cos(pitch)]
    ])
    
    R_2d = R_roll @ R_pitch

    rotation_inv, translation_inv = R_pitch.T, -translation

    imu_inv = np.identity(3)
    imu_inv[:2,:2] = rotation_inv
    imu_inv[:2,2] = translation_inv
    
    return imu_inv

def load_camera_params(cal): #c: calibration
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
    R_veh2cam = np.transpose(rotation_from_euler(roll, pitch, yaw))
    T_veh2cam = translation_matrix((-x+baseline/2, -y, -z))

    # Rotate to camera coordinates
    R = np.array([[0., -1., 0., 0.],
                  [0., 0., -1., 0.],
                  [1., 0., 0., 0.],
                  [0., 0., 0., 1.]])

    RT = R @ R_veh2cam @ T_veh2cam
    return RT, K

def generate_direct_backward_mapping(
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
            # world_coord : [world_x, world_y, 0, 1]
            # uv_coord : [u, v, 1]
            
            world_coord = [world_x, world_y, world_z, 1]
            camera_coord = extrinsic[:3, :] @ world_coord # 3*4 * 4*1 = 3*1
            uv_coord = intrinsic[:3, :3] @ camera_coord # 3*3 * 3*1 = 3*1
            uv_coord /= uv_coord[2]

            uv_coord = motion_cancel_mat @ uv_coord

            # map_x : (H, W)
            # map_y : (H, W)
            # dst[i][j] = src[ map_y[i][j] ][ map_x[i][j] ]
            map_x[i][j] = uv_coord[0]
            map_y[i][j] = uv_coord[1]
            
    return map_x, map_y

def quaternion_rotation_matrix(Q):
    q0, q1, q2, q3 = Q
    R = np.array([[2 * (q0 * q0 + q1 * q1) - 1,  2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
                            [2 * (q1 * q2 + q0 * q3), 2 * (q0 * q0 + q2 * q2) - 1, 2 * (q2 * q3 - q0 * q1)],
                            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 2 * (q0 * q0 + q3 * q3) - 1]])
                            
    return R

def euler_from_quaternion(q):
    x, y, z, w = q
    
    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # roll_x = math.atan2(t0, t1)

    # t2 = +2.0 * (w * y - z * x)
    # t2 = +1.0 if t2 > +1.0 else t2
    # t2 = -1.0 if t2 < -1.0 else t2
    # pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    matrix = [[math.cos(yaw), -math.sin(yaw)],
             [math.sin(yaw), math.cos(yaw)]]
    return matrix, yaw

def get_bbox_corners(center, dimensions, wx_interval, wy_interval):
    w, l, _ = dimensions
    x, y, _ = center

    l2, w2 = (l / 2)/wx_interval, (w / 2)/wy_interval

    corner = np.array([
        [x - l2, y - w2],  # Bottom-left
        [x + l2, y - w2],  # Bottom-right
        [x + l2, y + w2],  # Top-right
        [x - l2, y + w2],  # Top-left
    ])
    return corner

def rotate_bbox(center, corner, rotation_matrix):
    diff = []
    for corner_x, corner_y in corner:
        dx = corner_x - center[0]
        dy = corner_y - center[1]
        diff.append([dx, dy])
    diff = np.array(diff)
    rotated_corner = np.dot(rotation_matrix, diff.T).T
    rotated_corner = [[x+center[0], y+center[1]] for x, y in rotated_corner]
    return rotated_corner

def main():
    split = 'train'
    wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = 7, 70, 0.05, -10, 10, 0.025
    bev_height, bev_width = int((wx_max-wx_min)/wx_interval), int((wy_max-wy_min)/wy_interval)
    dir_download = '/home/jingjingtapa/다운로드'
    dir_image = f'{dir_download}/leftImg8bit/{split}'
    dir_calibration = f'{dir_download}/camera/{split}'
    dir_box = f'{dir_download}/gtBbox3d/{split}'
    city_list = os.listdir(dir_image)
    cnt_city, cnt_image = 0, 0

    image_list = sorted(os.listdir(f'{dir_image}/{city_list[cnt_city]}'))
    calibration_list = sorted(os.listdir(f'{dir_calibration}/{city_list[cnt_city]}'))
    box_list = sorted(os.listdir(f'{dir_box}/{city_list[cnt_city]}'))
    
    while True:
        image_dir = f'{dir_image}/{city_list[cnt_city]}/{image_list[cnt_image]}'
        calibration_dir = f'{dir_calibration}/{city_list[cnt_city]}/{calibration_list[cnt_image]}'
        box_dir = f'{dir_box}/{city_list[cnt_city]}/{box_list[cnt_image]}'

        print(image_dir)
        print(calibration_dir)
        print(box_dir)

        image, calibration, box = cv2.imread(image_dir), read_json(calibration_dir), read_json(box_dir)

        extrinsic, intrinsic = load_camera_params(calibration)

        motion_cancel_mat = motion_cancel(box)

        map_x, map_y = generate_direct_backward_mapping(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval, extrinsic, intrinsic, motion_cancel_mat)

        output_image = cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        # cv2.putText(output_image, image_dir, (10,10), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255) )

        for object in box["objects"]:
            center, type, quarternion, dimension = object['3d']['center'], object['3d']['type'], object['3d']['rotation'], object['3d']['dimensions']
            if center[0] >= wx_min and center[0] <= wx_max and center[1] >= wy_min and center[1] <= wy_max:
                # 이미지 하단 정중앙을 (0,0)으로 좌표 변환
                cx, cy = int(0.5*bev_width-center[1]/wy_interval), int(bev_height-center[0]/wx_interval)
                rotation_mat, yaw = euler_from_quaternion(quarternion)
                corner = get_bbox_corners(center= [cx,cy,1], dimensions =dimension, wx_interval= wx_interval, wy_interval= wy_interval)
                rotated_corner = rotate_bbox([cx, cy], corner,rotation_mat)

                cv2.circle(output_image, (cx, cy), 5, (255, 255, 0), -1, cv2.LINE_AA)
                for x,y in corner:
                    cv2.circle(output_image, (int(x), int(y)), 5, (0, 255, 0), -1, cv2.LINE_AA)
                
                for x,y in rotated_corner:
                    cv2.circle(output_image, (int(x), int(y)), 5, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.putText(output_image, f'{center}/{type}/yaw:{np.rad2deg(yaw)}', (cx-100, cy-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0) )

        cv2.imshow('BEV',output_image)
        cv2.imshow('original', image)
        key = cv2.waitKey(0)
        if key == ord('n') and cnt_image <= len(image_list) - 2:
            cnt_image = (cnt_image + 1) % len(image_list)
        elif key == ord('b') and cnt_image >= 1:
            cnt_image = (cnt_image - 1) % len(image_list)
        # elif key == ord('c') and cnt_city <= len(city_list) - 2:
        #     cnt_city = (cnt_city + 1) % len(city_list)
        elif key == ord('q'): 
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()