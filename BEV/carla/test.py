import os, getpass

username = getpass.getuser()
dir_download = f'/home/{username}/다운로드/dataset'

if not os.path.isdir(f'{dir_download}/normal'):
    os.makedirs(f'{dir_download}/normal')