import torch, getpass, os, csv
from BEVConverter import BEVConverter

def read_csv(file_path):
    data_list = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            data_list.append(row)
    return data_list

def load_img_cal_bbox(list):
    img_list, cal_list, bbox_list = [], [], []
    for scene in list:
        img = ''.join([scene[0],'_leftImg8bit.png'])
        cal = ''.join([scene[0],'_camera.json'])
        bbox = ''.join([scene[0],'_gtBbox3d.json'])
        img_list.append(img)
        cal_list.append(cal)
        bbox_list.append(bbox)
    return img_list, cal_list, bbox_list

class BEVDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        super(BEVDataset,self).__init__()
        username = getpass.getuser()
        
        self.dir_download = f'/home/{username}/다운로드'
        self.dir_image_folder, self.dir_calibration_folder, self.dir_gtbbox_folder = f'{self.dir_download}/leftImg8bit/{split}', f'{self.dir_download}/camera/{split}', f'{self.dir_download}/gtBbox3d/{split}'

        self.img_list, self.cal_list, self.bbox_list = load_img_cal_bbox(read_csv(f'{split}.csv'))
        
        self.BEVConverter = BEVConverter()

    def __getitem__(self,index):
        city = self.img_list[index].split("_")[0]

        img_dir = f'{self.dir_image_folder}/{city}/{self.img_list[index]}'
        cal_dir = f'{self.dir_calibration_folder}/{city}/{self.cal_list[index]}'
        bbox_dir = f'{self.dir_gtbbox_folder}/{city}/{self.bbox_list[index]}'

        bev_image, bev_objects = self.BEVConverter.get_BEV_image(img_dir, cal_dir, bbox_dir)
        return bev_image, bev_objects
    
    def __len__(self):    
        return len(self.x)
    