import getpass, os

def save_list_to_txt(data_list, file_path):
    with open(file_path, mode='w') as file:
        for row in data_list:
            file.write(' '.join(row) + '\n')

username = getpass.getuser()
dir_download = f'/home/{username}/다운로드'
dir_image, dir_calibration, dir_gtbbox = f'{dir_download}/leftImg8bit', f'{dir_download}/camera', f'{dir_download}/gtBbox3d'
split_list = os.listdir(dir_image)
for split in split_list:
    city_list = os.listdir(f'{dir_image}/{split}')
    scene_list = []
    for city in city_list:
        image_list = os.listdir(f'{dir_image}/{split}/{city}')
        image_list = sorted(image_list)
        for image in image_list:
            image = image.replace('_leftImg8bit.png', "")
            scene_list.append([image])
    save_list_to_txt(scene_list, f'{split}.txt')
