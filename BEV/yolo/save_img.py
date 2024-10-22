from cityscapesscripts.helpers.box3dImageTransform import Camera,Box3dImageTransform, CRS_V, CRS_C, CRS_S
from cityscapesscripts.helpers.annotation import CsBbox3d
from tqdm import tqdm
import os, getpass, json, math, cv2
from BEVConverter import BEVConverter
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
dir_img = f'{dir_download}/leftImg8bit'
dir_cal = f'{dir_download}/camera'
dir_box = f'{dir_download}/gtBbox3d'

split_list = os.listdir(dir_img)

if not os.path.isdir(f'{dir_download}/dataset'):
    os.makedirs(f'{dir_download}/dataset')

if not os.path.isdir(f'{dir_download}/dataset/images'):
    os.makedirs(f'{dir_download}/dataset/images')


wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = 7, 40, 0.05, -10, 10, 0.05
bev_height, bev_width = int((wx_max-wx_min)/wx_interval), int((wy_max-wy_min)/wy_interval)
bevconverter = BEVConverter(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval)
for split in tqdm(split_list):
    city_list = os.listdir(f'{dir_img}/{split}')
    
    if not os.path.isdir(f'{dir_download}/dataset/images/{split}'):
        os.makedirs(f'{dir_download}/dataset/images/{split}')

    scene_list = []
    for city in city_list:
        if not os.path.isdir(f'{dir_download}/dataset/images/{split}/{city}'):
            os.makedirs(f'{dir_download}/dataset/images/{split}/{city}')

        img_list = os.listdir(f'{dir_img}/{split}/{city}')
        img_list = sorted(img_list)

        for img in img_list:
            scene_name = img.split("_")[:3]
            scene_name = f'{scene_name[0]}_{scene_name[1]}_{scene_name[2]}'
            img_dir, cal_dir, bbox_dir = f'{dir_img}/{split}/{city}/{img}', f'{dir_cal}/{split}/{city}/{scene_name}_camera.json', f'{dir_box}/{split}/{city}/{scene_name}_gtBbox3d.json'
            bev_image, bev_objects = bevconverter.get_BEV_image(img_dir, cal_dir, bbox_dir)
            bev_img_save_dir = f'{dir_download}/dataset/images/{split}/{city}/{img}'
            cv2.imwrite(bev_img_save_dir, bev_image)
