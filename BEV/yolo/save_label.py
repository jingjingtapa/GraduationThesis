from cityscapesscripts.helpers.box3dImageTransform import Camera,Box3dImageTransform, CRS_V, CRS_C, CRS_S
from cityscapesscripts.helpers.annotation import CsBbox3d
from tqdm import tqdm
import os, getpass, json, math

def read_json(dir):
    with open(dir, 'r') as file:
        data = json.load(file)
    return data

def save_list_to_txt(data_list, file_path):
    with open(file_path, mode='w') as file:
        for row in data_list:
            row = [str(item) for item in row]
            file.write(' '.join(row) + '\n')

username = getpass.getuser()
dir_download = f'/home/{username}/다운로드'
dir_gtbbox = f'{dir_download}/gtBbox3d'
split_list = os.listdir(dir_gtbbox)

if not os.path.isdir(f'{dir_download}/dataset/train'):
    os.makedirs(f'{dir_download}/dataset/train')
if not os.path.isdir(f'{dir_download}/dataset/test'):
    os.makedirs(f'{dir_download}/dataset/test')
if not os.path.isdir(f'{dir_download}/dataset/val'):
    os.makedirs(f'{dir_download}/dataset/val')

wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = 7, 40, 0.05, -10, 10, 0.05
bev_height, bev_width = int((wx_max-wx_min)/wx_interval), int((wy_max-wy_min)/wy_interval)

for split in tqdm(split_list):

    city_list = os.listdir(f'{dir_gtbbox}/{split}')
    
    if not os.path.isdir(f'{dir_download}/dataset/{split}/images'):
        os.makedirs(f'{dir_download}/dataset/{split}/images')

    if not os.path.isdir(f'{dir_download}/dataset/{split}/labels'):
        os.makedirs(f'{dir_download}/dataset/{split}/labels')

    scene_list = []
    for city in city_list:
        if not os.path.isdir(f'{dir_download}/dataset/{split}/images/{city}'):
            os.makedirs(f'{dir_download}/dataset/{split}/images/{city}')

        if not os.path.isdir(f'{dir_download}/dataset/{split}/labels/{city}'):
            os.makedirs(f'{dir_download}/dataset/{split}/labels/{city}')

        gtbbox_list = os.listdir(f'{dir_gtbbox}/{split}/{city}')
        gtbbox_list = sorted(gtbbox_list)
        for gtbbox in gtbbox_list:
            gtbbox_dir = f'{dir_gtbbox}/{split}/{city}/{gtbbox}'
            data = read_json(gtbbox_dir)
            scene_name = gtbbox.replace('_gtBbox3d.json','_leftImg8bit')
            camera = Camera(fx=data["sensor"]["fx"],fy=data["sensor"]["fy"],
            u0=data["sensor"]["u0"],v0=data["sensor"]["v0"],
            sensor_T_ISO_8855=data["sensor"]["sensor_T_ISO_8855"])

            box3d_annotation = Box3dImageTransform(camera=camera)

            obj = CsBbox3d()
            objects = []
            for object in data['objects']:
                obj.fromJsonText(object)
                box3d_annotation.initialize_box_from_annotation(obj, coordinate_system=CRS_V)
                box_vertices_C = box3d_annotation.get_vertices(coordinate_system=CRS_C)
                bottom_vertice = []
                for loc, pv_coord in box_vertices_C.items():
                    if loc in ['FLB', 'FRB', 'BLB', 'BRB']:
                        vx, vy = pv_coord[0], pv_coord[1]
                        u, v = int(0.5*bev_width - vy/wy_interval), int(bev_height - (vx - wx_min)/wx_interval)
                        nu, nv = round(u/bev_width,3), round(v/bev_height,3)
                        # bottom_vertice.append(round(u,3))
                        # bottom_vertice.append(round(v,3))
                        if 0<= nu <1 and 0<= nv<1:
                            bottom_vertice.append(nu)
                            bottom_vertice.append(nv)
                # class x_center y_center width height
                if object['label'] == 'car': 
                    cls = 1
                else:
                    cls = 0
                object_spec = [cls, *bottom_vertice]  
                
                if len(object_spec) == 9:
                    objects.append([cls, *bottom_vertice])
            
            save_list_to_txt(objects,f'{dir_download}/dataset/{split}/labels/{city}/{scene_name}.txt')