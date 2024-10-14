import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import BEVDataset

import matplotlib.pyplot as plt

def show_bev_image(dataset, index):
    bev_image, bev_objects = dataset[index]   

    plt.imshow(bev_image)
    plt.title(f"BEV Image {index}")
    plt.show()

wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = 7, 40, 0.05, -10, 10, 0.05
train_dataset = BEVDataset('train',wx_min = wx_min,wx_max = wx_max, wx_interval=wx_interval,
                            wy_min=wy_min,wy_max= wy_max,wy_interval= wy_interval)

show_bev_image(train_dataset, 0)