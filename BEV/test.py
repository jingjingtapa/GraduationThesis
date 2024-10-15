from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(data='coco8.yaml',epochs = 100,  imgsz = 640)