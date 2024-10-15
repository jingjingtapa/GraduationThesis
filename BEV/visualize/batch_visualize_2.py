from cityscapesscripts.helpers.box3dImageTransform import Camera,Box3dImageTransform, CRS_V, CRS_C, CRS_S
from cityscapesscripts.helpers.annotation import CsBbox3d
import cv2, json, os, math
import numpy as np

def read_yolo_labels(label_file_path):
    labels = []

    with open(label_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = list(map(float, line.strip().split()))
            class_id = int(values[0])
            x1, y1, x2, y2, x3, y3, x4, y4 = values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]
            labels.append([class_id, x1, y1, x2, y2, x3, y3, x4, y4])
    
    return labels

def main():
    split = 'train'
    wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = 7, 40, 0.05, -10, 10, 0.05
    bev_height, bev_width = int((wx_max-wx_min)/wx_interval), int((wy_max-wy_min)/wy_interval)
    dir_download = '/home/jingjingtapa/다운로드'
    dir_image = f'{dir_download}/dataset/{split}/images'
    dir_label = f'{dir_download}/dataset/{split}/labels'
    city_list = os.listdir(dir_image)
    cnt_city, cnt_image = 0, 0

    image_list = sorted(os.listdir(f'{dir_image}/{city_list[cnt_city]}'))
    label_list = sorted(os.listdir(f'{dir_label}/{city_list[cnt_city]}'))
    
    while True:
        image_dir = f'{dir_image}/{city_list[cnt_city]}/{image_list[cnt_image]}'
        label_dir = f'{dir_label}/{city_list[cnt_city]}/{label_list[cnt_image]}'
        print(image_dir)
        bev_image = cv2.imread(image_dir)
        label = read_yolo_labels(label_dir)
        print(label)
        for cls, x1, y1, x2, y2, x3, y3, x4, y4 in label:
            
            cv2.circle(bev_image, (int(x1), int(y1)), 2, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(bev_image, (int(x2), int(y2)), 2, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(bev_image, (int(x3), int(y3)), 2, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(bev_image, (int(x4), int(y4)), 2, (255, 255, 0), -1, cv2.LINE_AA)

        cv2.imshow('BEV',bev_image)
        # cv2.imshow('original', image)
        key = cv2.waitKey(0)
        if key == ord('n') and cnt_image <= len(image_list) - 2:
            cnt_image = (cnt_image + 1) % len(image_list)
        elif key == ord('b') and cnt_image >= 1:
            cnt_image = (cnt_image - 1) % len(image_list)
        elif key == ord('c') and cnt_city <= len(city_list) - 2:
            cnt_city = (cnt_city + 1) % len(city_list)
            cnt_image = 0
        elif key == ord('q'): 
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()