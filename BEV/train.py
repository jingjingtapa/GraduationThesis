import ultralytics, torch
from ultralytics import YOLO

torch.backends.cudnn.enabled = False

ultralytics.checks()

model = YOLO('yolo11n-obb.pt')

wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = 7, 40, 0.05, -10, 10, 0.05

model.train(data = 'dataset.yaml', epochs = 5, imgsz=640)