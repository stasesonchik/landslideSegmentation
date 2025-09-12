# train.py
import os
import time
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import rasterio
import segmentation_models_pytorch as smp

from dataset import CanopyDataset, parse_annotations, get_transforms, IMG_SIZE
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)



# --- Конфигурация путей
PATH = "C:/Users/LeMeS/PycharmProjects/Competition"
TRAIN_IMG_DIR = os.path.join(PATH, "train")
TRAIN_ANNOT_PATH = os.path.join(PATH, "train_annotations.json")
OUTPUT_DIR = os.path.join(PATH, "runs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Гиперпараметры
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PERIOD = 5
NUM_CLASSES = 3  # фон + 2 класса

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_model(encoder_name="resnet34"):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES,
    )
    return model

# --- IoU по классам (без фона)
def iou_score(preds, target, num_classes=NUM_CLASSES):
    ious = []
    for cls in range(1, num_classes):
        p = (preds == cls).astype(np.uint8)
        t = (target == cls).astype(np.uint8)
        inter = (p & t).sum()
        union = (p | t).sum()
        if union == 0:
            iou = 1.0
        else:
            iou = inter / union
        ious.append(iou)
    return float(np.mean(ious))

def train():
    seed_everything(42)

    # --- Загружаем аннотации
    mapping = parse_annotations(TRAIN_ANNOT_PATH)
    file_list = sorted(list(mapping.keys()))
    split = int(0.9 * len(file_list))
    train_files = set(file_list[:split])
    val_files = set(file_list[split:])

    train_map = {k: v for k, v in mapping.items() if k in train_files}
    val_map = {k: v for k, v in mapping.items() if k in val_files}

    train_ds = CanopyDataset(TRAIN_IMG_DIR, train_map, transforms=get_transforms(IMG_SIZE, train=True), is_test=False)
    val_ds = CanopyDataset(TRAIN_IMG_DIR, val_map, transforms=get_transforms(IMG_SIZE, train=False), is_test=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    model = build_model().to(DEVICE)

    # --- Loss
    dice = smp.losses.DiceLoss(mode='multiclass')
    ce = nn.CrossEntropyLoss()
    def criterion(outputs, masks):
        return dice(outputs, masks) + ce(outputs, masks)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # --- Для графиков
    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    best_val_iou = -1.0
    best_path = None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Train E{epoch}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validation
        model.eval()
        val_loss = 0.0
        val_ious = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Val"):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()
                gt = masks.squeeze(0).cpu().numpy()
                val_ious.append(iou_score(pred, gt))

        val_loss /= len(val_loader.dataset)
        mean_val_iou = float(np.mean(val_ious)) if val_ious else 0.0
        scheduler.step(val_loss)

        # --- Сохраняем статистику
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(mean_val_iou)

        print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_iou: {mean_val_iou:.4f} | time: {time.time()-t0:.1f}s")

        # --- Сохраняем модель каждые SAVE_PERIOD эпох
        if epoch % SAVE_PERIOD == 0:
            path = os.path.join(OUTPUT_DIR, f"unet_resnet34_epoch{epoch}_iou{mean_val_iou:.4f}.pth")
            torch.save(model.state_dict(), path)
            print(f"Saved model at epoch {epoch} -> {path}")

        # --- Сохраняем лучшую модель
        if mean_val_iou > best_val_iou:
            best_val_iou = mean_val_iou
            best_path = os.path.join(OUTPUT_DIR, f"best_unet_resnet34_epoch{epoch}_iou{best_val_iou:.4f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model -> {best_path}")

    # --- Построение графиков
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.plot(history["train_loss"], label="train_loss"); plt.plot(history["val_loss"], label="val_loss"); plt.legend(); plt.title("Loss")
    plt.subplot(1,3,2); plt.plot(history["val_iou"], label="val_iou"); plt.legend(); plt.title("Mean IoU")
    plt.tight_layout()
    plt.show()

    print("Training finished. Best:", best_path)

if __name__ == "__main__":
    train()
