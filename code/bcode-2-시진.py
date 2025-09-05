
import os
import time
import random

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import albumentations as A
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 시드를 고정합니다.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# 데이터셋 클래스를 정의합니다.
class ImageDataset(Dataset):
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target

# 데이터셋 클래스를 정의합니다.
class ImageDataset2(Dataset):
    def __init__(self, df, path, transform=None):
        #self.df = pd.read_csv(csv).values
        self.df = df.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, target

# one epoch 학습을 위한 함수입니다.
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()

        #model.zero_grad(set_to_none=True)

        with autocast():
            preds = model(image)
            loss = loss_fn(preds, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

    return ret

def validate_one_epoch(loader, model, loss_fn, device):
    model.eval()
    val_loss = 0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for image, targets in tqdm(loader):
            image = image.to(device)
            targets = targets.to(device)

            with autocast():
                preds = model(image)
                loss = loss_fn(preds, targets)

            val_loss += loss.item()
            preds_list.extend(preds.argmax(dim=1).cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    val_loss /= len(loader)
    val_acc = accuracy_score(targets_list, preds_list)
    val_f1 = f1_score(targets_list, preds_list, average='macro')

    return {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
    }


# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data config
#data_path = 'datasets_fin/'
data_path = '../'

# model config
#model_name = 'resnet34' # 'resnet50' 'efficientnet-b0', ...
#model_name = 'tf_efficientnet_b4_ns'
#model_name = 'tf_efficientnet_b8'
#model_name = 'tf_efficientnet_b4.ns_jft_in1k'
model_name = 'tf_efficientnet_b6.aa_in1k'
#model_name = 'tf_efficientnet_b5.ns_jft_in1k'

# training config
#img_size = 32
#img_size = 224
img_size = 528
#img_size = 456
LR = 1e-3
#EPOCHS = 20
EPOCHS = 100
#EPOCHS = 10
#EPOCHS = 40
#EPOCHS = 15
#BATCH_SIZE = 32
BATCH_SIZE = 16
num_workers = 0

# augmentation을 위한 transform 코드
#trn_transform = A.Compose([
#    # 이미지 크기 조정
#    A.Resize(height=img_size, width=img_size),
#    # images normalization
#    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#    # numpy 이미지나 PIL 이미지를 PyTorch 텐서로 변환
#    ToTensorV2(),
#])
#trn_transform = A.Compose([
#    A.Resize(img_size, img_size),
#    A.RandomBrightnessContrast(p=0.5),
#    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=5, p=0.7),
#    A.MotionBlur(p=0.2),
#    A.OpticalDistortion(p=0.2),
#    A.Cutout(max_h_size=int(img_size*0.1), max_w_size=int(img_size*0.1), num_holes=5, p=0.5),
#    A.Normalize(mean=[0.485, 0.456, 0.406],
#                std=[0.229, 0.224, 0.225]),
#    ToTensorV2(),
#])
trn_transform = A.Compose([
    A.Resize(img_size, img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.ColorJitter(p=0.4),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),

    A.MotionBlur(p=0.2),
    A.OpticalDistortion(p=0.2),
    A.Cutout(max_h_size=int(img_size*0.1), max_w_size=int(img_size*0.1), num_holes=5, p=0.5),

    A.GridDistortion(p=0.3),
    A.CoarseDropout(max_holes=8, max_height=img_size//10, max_width=img_size//10, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# test image 변환을 위한 transform 코드
tst_transform = A.Compose([
    A.Resize(height=img_size, width=img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
#tst_transform = A.Compose([
#    A.Resize(img_size, img_size),
#    A.RandomBrightnessContrast(p=0.5),
#    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=5, p=0.7),
#    A.MotionBlur(p=0.2),
#    A.OpticalDistortion(p=0.2),
#    A.Cutout(max_h_size=int(img_size*0.1), max_w_size=int(img_size*0.1), num_holes=5, p=0.5),
#    A.Normalize(mean=[0.485, 0.456, 0.406],
#                std=[0.229, 0.224, 0.225]),
#    ToTensorV2(),
#])

# Dataset 정의
#trn_dataset = ImageDataset(
#    "datasets_fin/train.csv",
#    "datasets_fin/train/",
#    transform=trn_transform
#)
#tst_dataset = ImageDataset(
#    "datasets_fin/sample_submission.csv",
#    "datasets_fin/test/",
#    transform=tst_transform
#)
#trn_dataset = ImageDataset(
#    "../train.csv",
#    "../train/",
#    transform=trn_transform
#)
df = pd.read_csv("../train.csv")

# stratify를 통해 클래스 비율 유지
train_df, val_df = train_test_split(
    df,
    #test_size=0.2,
    test_size=0.1,
    stratify=df['target'],
    random_state=SEED
)

trn_dataset = ImageDataset2(train_df, "../train/", transform=trn_transform)
val_dataset = ImageDataset2(val_df, "../train/", transform=tst_transform)


tst_dataset = ImageDataset(
    "../sample_submission.csv",
    "../test/",
    transform=tst_transform
)
print(len(trn_dataset), len(tst_dataset))

# DataLoader 정의
trn_loader = DataLoader(
    trn_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

tst_loader = DataLoader(
    tst_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# load model
model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes=17
).to(device)
loss_fn = nn.CrossEntropyLoss()
#optimizer = Adam(model.parameters(), lr=LR)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

#for epoch in range(EPOCHS):
#    ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device=device)
#    ret['epoch'] = epoch
#
#    log = ""
#    for k, v in ret.items():
#      log += f"{k}: {v:.4f}\n"
#    print(log)

best_f1 = 0
for epoch in range(EPOCHS):
    train_ret = train_one_epoch(trn_loader, model, optimizer, loss_fn, device=device)
    val_ret = validate_one_epoch(val_loader, model, loss_fn, device=device)  # 따로 구현 필요

    if val_ret['val_f1'] > best_f1:
        print(f"Epoch {epoch} - val_f1: {val_ret['val_f1']:.4f}")

        best_f1 = val_ret['val_f1']
        torch.save(model.state_dict(), 'model/best_model2.pth')

# 저장된 가중치 로드
model.load_state_dict(torch.load('model/best_model2.pth'))

preds_list = []

model.eval()
for image, _ in tqdm(tst_loader):
    image = image.to(device)

    with torch.no_grad():
        preds = model(image)
    preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())

pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = preds_list

#sample_submission_df = pd.read_csv("datasets_fin/sample_submission.csv")
sample_submission_df = pd.read_csv("../sample_submission.csv")
assert (sample_submission_df['ID'] == pred_df['ID']).all()

pred_df.to_csv("pred-2.csv", index=False)

print("pred-2.csv saved!!")
