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

warnings.filterwarnings("ignore")
DEVICE = CFG.DEVICE
NUM_CLASSES = len(CFG.CLASS_NAMES) + 1  # +1 для background
class_to_id = {name: i + 1 for i, name in enumerate(CFG.CLASS_NAMES)}

EPOCHS = 12  # уменьшено для маленького датасета


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
    freeze = True
    for name, param in model.encoder.named_parameters():
        if freeze:
            param.requires_grad = False
        if freeze_until in name:
            freeze = False
    return model.to(DEVICE)


# -------------------------
class WeightedCombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, masks):
        ce_loss = self.ce(outputs, masks)
        dice_loss = self.dice(outputs, masks)
        return dice_loss + ce_loss


# -------------------------
def iou_score(preds, target):
    preds, target = preds.detach().cpu().numpy(), target.detach().cpu().numpy()
    ious = []
    for cls in range(1, NUM_CLASSES):
        p = (preds == cls).astype(np.uint8)
        t = (target == cls).astype(np.uint8)
        inter = (p & t).sum()
        union = (p | t).sum()
        ious.append(1.0 if union == 0 else inter / union)
    return ious  # по классам


# -------------------------
def map_at_iou(preds, target, iou_thresh=0.75):
    preds, target = preds.detach().cpu().numpy(), target.detach().cpu().numpy()
    class_matches = []
    for cls in range(1, NUM_CLASSES):
        p_mask = (preds == cls).astype(np.uint8)
        t_mask = (target == cls).astype(np.uint8)
        inter = (p_mask & t_mask).sum()
        union = (p_mask | t_mask).sum()
        match = 1 if union > 0 and inter / union >= iou_thresh else 0
        class_matches.append(match)
    return np.mean(class_matches), class_matches  # общий mAP@0.75 и по классам


# -------------------------
def get_transforms(trial):
    train_tf = A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.HorizontalFlip(p=trial.suggest_float("hflip_p", 0, 1)),
        A.VerticalFlip(p=trial.suggest_float("vflip_p", 0, 1)),
        A.RandomRotate90(p=trial.suggest_float("r90_p", 0, 1)),
        A.Affine(
            translate_percent=(-trial.suggest_float("trans", 0, 0.2), trial.suggest_float("trans", 0, 0.2)),
            scale=(1 - trial.suggest_float("scale", 0, 0.2), 1 + trial.suggest_float("scale", 0, 0.2)),
            rotate=(-trial.suggest_int("rotate", 0, 30), trial.suggest_int("rotate", 0, 30))
        ),
        A.HueSaturationValue(p=trial.suggest_float("hsv_p", 0, 1)),
        A.GaussianBlur(p=trial.suggest_float("blur_p", 0, 0.5)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_tf, val_tf


# -------------------------
def objective(trial, resolution="20"):
    df = parse_annotations(CFG.TRAIN_ANNOT_PATH)
    # фильтр по разрешению, если необходимо
    df_res = df[df['cm_resolution'] == int(resolution)]

    train_tf, val_tf = get_transforms(trial)
    dataset = CanopyDataset(CFG.TRAIN_IMG_PATH, df_res, class_to_id, transforms=train_tf, return_filename=True)

    train_size = int(0.9     * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transforms = val_tf

    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model("resnet18", freeze_until="layer2")
    criterion = WeightedCombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_map = -1.0

    for epoch in range(EPOCHS):
        # ===== TRAIN =====
        model.train()
        epoch_loss = 0.0
        for images, masks, _ in tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", leave=False):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
        train_loss = epoch_loss / len(train_loader.dataset)

        # ===== VALID =====
        model.eval()
        val_loss = 0.0
        val_maps = []
        classwise_maps_accum = np.zeros(NUM_CLASSES - 1)
        classwise_iou_accum = np.zeros(NUM_CLASSES - 1)
        counts = np.zeros(NUM_CLASSES - 1)

        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc=f"Val Epoch {epoch + 1}", leave=False):
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)

                batch_map, class_map = map_at_iou(preds, masks)
                val_maps.append(batch_map)
                classwise_maps_accum += np.array(class_map)
                classwise_iou_accum += np.array(iou_score(preds, masks))
                counts += 1

        val_loss /= len(val_loader.dataset)
        val_map_mean = np.mean(val_maps)
        classwise_map_mean = classwise_maps_accum / counts
        classwise_iou_mean = classwise_iou_accum / counts

        print(
            f"Epoch {epoch + 1} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | mAP@0.75: {val_map_mean:.4f}")
        for cls_name, map_cls, iou_cls in zip(CFG.CLASS_NAMES, classwise_map_mean, classwise_iou_mean):
            print(f"  Class {cls_name}: mAP@0.75={map_cls:.4f}, IoU={iou_cls:.4f}")

        if val_map_mean > best_val_map:
            best_val_map = val_map_mean
            trial_save_path = os.path.join(CFG.SAVE_PATH, f"trial_{trial.number}_{resolution}cm")
            os.makedirs(trial_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(trial_save_path, "best_model.pth"))

    print(f"Trial {trial.number} ({resolution}cm) finished | Best mAP@0.75: {best_val_map:.4f}")
    return -best_val_map


# -------------------------
if __name__ == "__main__":
    seed_everything()
    os.makedirs(CFG.SAVE_PATH, exist_ok=True)

    resolutions = ["20", "40", "60", "80"]
    for res in resolutions:
        print(f"\n=== Optuna for {res}cm ===")
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, resolution=res), n_trials=5)
        print(f"Best trial params for {res}cm:", study.best_trial.params)
