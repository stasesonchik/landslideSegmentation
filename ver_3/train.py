import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import optuna
import warnings
from tqdm import tqdm
from dataset import CanopyDataset, parse_annotations, CFG
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DEVICE = CFG.DEVICE
NUM_CLASSES = len(CFG.CLASS_NAMES) + 1
class_to_id = {name: i+1 for i, name in enumerate(CFG.CLASS_NAMES)}

EPOCHS = 30  # Для Optuna меньше

# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------
def build_model(encoder_name="resnet18", freeze_until="layer2"):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    )
    # заморозка encoder
    freeze = True
    for name, param in model.encoder.named_parameters():
        if freeze:
            param.requires_grad = False
        if freeze_until in name:
            freeze = False
    return model.to(DEVICE)

# -------------------------
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.ce = nn.CrossEntropyLoss()
    def forward(self, outputs, masks):
        return self.dice(outputs, masks) + self.ce(outputs, masks)

# -------------------------
def iou_score(preds, target, num_classes=NUM_CLASSES):
    preds, target = preds.detach().cpu().numpy(), target.detach().cpu().numpy()
    ious = []
    for cls in range(1, num_classes):
        p = (preds==cls).astype(np.uint8)
        t = (target==cls).astype(np.uint8)
        inter = (p&t).sum()
        union = (p|t).sum()
        ious.append(1.0 if union==0 else inter/union)
    return float(np.mean(ious)), ious

# -------------------------
def map_at_iou(preds, target, iou_thresh=0.75, num_classes=NUM_CLASSES):
    preds, target = preds.detach().cpu().numpy(), target.detach().cpu().numpy()
    matches_per_class, totals = [], []
    for cls in range(1, num_classes):
        p_mask=(preds==cls).astype(np.uint8)
        t_mask=(target==cls).astype(np.uint8)
        inter=(p_mask&t_mask).sum()
        union=(p_mask|t_mask).sum()
        match = 1 if union>0 and inter/union>=iou_thresh else 0
        matches_per_class.append(match)
        totals.append(1)
    return np.mean(matches_per_class), matches_per_class

# -------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    epoch_loss=0.0
    for images,masks in loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*images.size(0)
    return epoch_loss/len(loader.dataset)

# -------------------------
def validate_one_epoch(model, loader, criterion):
    model.eval()
    val_loss=0.0
    val_map=0.0
    val_ious=[]
    classwise_iou_sum = np.zeros(NUM_CLASSES-1)
    classwise_counts = np.zeros(NUM_CLASSES-1)
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            val_loss += criterion(outputs, masks).item()*images.size(0)
            preds = torch.argmax(outputs, dim=1)
            batch_map, batch_class_matches = map_at_iou(preds, masks)
            val_map += batch_map
            _, batch_class_iou = iou_score(preds, masks)
            val_ious.append(np.mean(batch_class_iou))
            classwise_iou_sum += np.array(batch_class_iou)
            classwise_counts += 1
    val_loss /= len(loader.dataset)
    val_map /= len(loader)
    val_ious_mean = np.mean(val_ious)
    classwise_iou_mean = classwise_iou_sum / classwise_counts
    return val_loss, val_map, val_ious_mean, classwise_iou_mean

# -------------------------
def plot_metrics(train_losses, val_losses, val_ious, val_maps):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(val_ious,label="val_iou")
    plt.title("IoU")
    plt.legend()
    plt.subplot(1,3,3)
    plt.plot(val_maps,label="val_map@0.75")
    plt.title("mAP@0.75")
    plt.legend()
    plt.show()

# -------------------------
# ------------------------- ОБНОВЛЁННЫЙ objective с логированием -------------------------
def objective(trial):
    # гиперпараметры аугментации
    train_tf = A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.HorizontalFlip(p=trial.suggest_float("hflip_p",0,1)),
        A.VerticalFlip(p=trial.suggest_float("vflip_p",0,1)),
        A.RandomRotate90(p=trial.suggest_float("r90_p",0,1)),
        A.Affine(
            translate_percent=(-trial.suggest_float("trans",0,0.2), trial.suggest_float("trans",0,0.2)),
            scale=(1-trial.suggest_float("scale",0,0.2), 1+trial.suggest_float("scale",0,0.2)),
            rotate=(-trial.suggest_int("rotate",0,30), trial.suggest_int("rotate",0,30))
        ),
        A.HueSaturationValue(p=trial.suggest_float("hsv_p",0,1)),
        A.GaussianBlur(p=trial.suggest_float("blur_p",0,0.5)),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

    # encoder и lr/batch_size
    encoder = trial.suggest_categorical("encoder", ["resnet18"])
    lr = trial.suggest_float("lr",1e-5,1e-3,log=True)
    batch_size = trial.suggest_categorical("batch_size",[2,4,8])

    print(f"\n=== Trial {trial.number} ===")
    print(f"Encoder: {encoder}, lr: {lr:.1e}, batch_size: {batch_size}")
    print("Augmentation parameters:")
    print({k: v for k, v in trial.params.items() if k not in ["encoder", "lr", "batch_size"]})

    df = parse_annotations(CFG.TRAIN_ANNOT_PATH)
    dataset = CanopyDataset(CFG.TRAIN_IMG_PATH, df, class_to_id, transforms=train_tf)
    train_size = int(0.8*len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset,[train_size,val_size])
    val_ds.dataset.transforms = val_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(encoder, freeze_until="layer2")
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_ious, val_maps = [], [], [], []
    best_val_map = -1.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        # прогресс по батчам
        model.train()
        epoch_loss = 0.0
        for images, masks in tqdm(train_loader, desc="Training", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*images.size(0)
        train_losses.append(epoch_loss/len(train_loader.dataset))

        # валидация
        model.eval()
        val_loss, val_map, val_iou, classwise_iou = 0,0,0,[0]*len(CFG.CLASS_NAMES)
        val_loss_acc, val_map_acc, val_ious_acc = [],[],[]
        classwise_iou_sum = np.zeros(NUM_CLASSES-1)
        classwise_counts = np.zeros(NUM_CLASSES-1)

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation", leave=False):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()*images.size(0)
                preds = torch.argmax(outputs, dim=1)
                batch_map, _ = map_at_iou(preds, masks)
                _, batch_class_iou = iou_score(preds, masks)
                val_map += batch_map
                val_ious_acc.append(np.mean(batch_class_iou))
                classwise_iou_sum += np.array(batch_class_iou)
                classwise_counts += 1

        val_loss /= len(val_loader.dataset)
        val_map /= len(val_loader)
        val_ious_mean = np.mean(val_ious_acc)
        classwise_iou_mean = classwise_iou_sum / classwise_counts

        val_losses.append(val_loss)
        val_ious.append(val_ious_mean)
        val_maps.append(val_map)

        print(f"train_loss: {train_losses[-1]:.4f} | val_loss: {val_loss:.4f} | val_iou: {val_ious_mean:.4f} | val_map@0.75: {val_map:.4f}")
        print(f"Classwise IoU: {dict(zip(CFG.CLASS_NAMES,classwise_iou_mean))}")

        if val_map > best_val_map:
            best_val_map = val_map
            torch.save(model.state_dict(), os.path.join(CFG.SAVE_PATH,"best_model.pth"))
            print("-> Saved best model!")

    print(f"\nTrial {trial.number} finished | Best mAP@0.75: {best_val_map:.4f}")
    return -best_val_map  # минимизируем для Optuna


# -------------------------
if __name__=="__main__":
    seed_everything()
    os.makedirs(CFG.SAVE_PATH, exist_ok=True)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    print("Best trial params:", study.best_trial.params)
