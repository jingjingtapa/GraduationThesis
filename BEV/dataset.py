import torch, getpass
from BEVConverter import BEVConverter

def read_txt(file_path):
    data_list = []
    with open(file_path, mode='r') as file:
        for line in file:
            data_list.append(line.strip())
    
    return data_list

def load_img_cal_bbox(list):
    img_list, cal_list, bbox_list = [], [], []
    for scene in list:
        img = ''.join([scene,'_leftImg8bit.png'])
        cal = ''.join([scene,'_camera.json'])
        bbox = ''.join([scene,'_gtBbox3d.json'])
        img_list.append(img)
        cal_list.append(cal)
        bbox_list.append(bbox)
    return img_list, cal_list, bbox_list

class BEVDataset(torch.utils.data.Dataset):
    def __init__(self, split, wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval):
        super(BEVDataset,self).__init__()
        username = getpass.getuser()
        
        self.dir_download = f'/home/{username}/다운로드'
        self.dir_image_folder, self.dir_calibration_folder, self.dir_gtbbox_folder = f'{self.dir_download}/leftImg8bit/{split}', f'{self.dir_download}/camera/{split}', f'{self.dir_download}/gtBbox3d/{split}'

        self.img_list, self.cal_list, self.bbox_list = load_img_cal_bbox(read_txt(f'{split}.txt'))
    
        self.BEVConverter = BEVConverter(wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval)

    def __getitem__(self,index):
        city = self.img_list[index].split("_")[0]

        img_dir = f'{self.dir_image_folder}/{city}/{self.img_list[index]}'
        cal_dir = f'{self.dir_calibration_folder}/{city}/{self.cal_list[index]}'
        bbox_dir = f'{self.dir_gtbbox_folder}/{city}/{self.bbox_list[index]}'

        bev_image, bev_objects = self.BEVConverter.get_BEV_image(img_dir, cal_dir, bbox_dir)
        bev_image, bev_objects = torch.from_numpy(bev_image).float(), torch.Tensor(bev_objects).float()
        return bev_image, bev_objects
    
    def __len__(self):    
        return len(self.img_list)
    