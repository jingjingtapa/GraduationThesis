import ultralytics, torch
from ultralytics import YOLO

torch.backends.cudnn.enabled = False

ultralytics.checks()

model = YOLO('yolo11n-obb.pt')

model.train(data = 'dataset.yaml', epochs = 100, imgsz=640, optimizer = 'SGD', lr0 = 0.001)