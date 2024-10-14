import torch
from torchvision import models, transforms
from ultralytics import YOLO
from dataset import BEVDataset
from torch.utils.data import DataLoader



wx_min, wx_max, wx_interval, wy_min, wy_max, wy_interval = 7, 40, 0.05, -10, 10, 0.05

train_dataset = BEVDataset('train',wx_min = wx_min,wx_max = wx_max, wx_interval=wx_interval,
                            wy_min=wy_min,wy_max= wy_max,wy_interval= wy_interval)
val_dataset = BEVDataset('val',wx_min = wx_min,wx_max = wx_max, wx_interval=wx_interval,
                            wy_min=wy_min,wy_max= wy_max,wy_interval= wy_interval)

sample_img, _ = train_dataset[0]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

classes = ['other','car']
# model = YOLO('yolov8s.pt')
model = models.detection.ssd300_vgg16(pretrained=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = [{  # SSD에 맞는 라벨 형식으로 변환
            'boxes': torch.tensor([[obj[1], obj[2], obj[3], obj[4]] for obj in targets], dtype=torch.float32).to(device),
            'labels': torch.tensor([obj[0] for obj in targets], dtype=torch.int64).to(device)
        }]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)  # 모델의 출력 및 손실 계산
        losses = sum(loss for loss in loss_dict.values())
        
        losses.backward()
        optimizer.step()
        running_loss += losses.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")



