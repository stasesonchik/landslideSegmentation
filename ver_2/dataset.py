# dataset.py
import os
import json
import glob
from pathlib import Path

import numpy as np
import rasterio
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict

# Конфиг (скорее минимальный — можно вынести в отдельный файл)
IMG_SIZE = 1024
CLASS_NAMES = ["individual_tree", "group_of_trees"]
# индекс класса: 1 и 2 (0 — фон)
class_to_id = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}

def parse_annotations(annot_path):
    """Читает train_annotations.json -> возвращает dict: file_name -> list of annots (class, segmentation)."""
    with open(annot_path, 'r') as f:
        data = json.load(f)
    images = data.get('images', [])
    mapping = defaultdict(list)
    for img in images:
        fn = img['file_name']
        for ann in img.get('annotations', []):
            mapping[fn].append({
                'class': ann['class'],
                'segmentation': ann['segmentation']
            })
    return mapping

def polygons_to_mask(height, width, annots, class_to_id_map):
    """Создаёт маску HxW с целыми значениями (0..C). annots — список {'class','segmentation'}."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for a in annots:
        cls = a['class']
        coords = a['segmentation']
        if coords is None or len(coords) < 6:
            continue
        pts = np.array(coords, dtype=np.int32).reshape(-1, 2)
        cid = class_to_id_map.get(cls)
        if cid is None:
            continue
        # cv2.fillPoly expects list of arrays
        cv2.fillPoly(mask, [pts], color=int(cid))
    return mask

def get_transforms(img_size=IMG_SIZE, train=True):
    if train:
        return A.Compose([
            A.RandomCrop(int(img_size*0.9), int(img_size*0.9)),
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

class CanopyDataset(Dataset):
    """
    Dataset читает .tif через rasterio и генерирует маску из аннотаций (если df_mapping предоставлен).
    Если is_test=True, возвращает (image_tensor, file_name, (original_h, original_w))
    Иначе: (image_tensor, mask_tensor) где mask_tensor содержит целые метки (0..C)
    """
    def __init__(self, image_dir, annot_mapping=None, transforms=None, is_test=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        self.annot_mapping = annot_mapping or {}
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        file_name = os.path.basename(img_path)
        with rasterio.open(img_path) as src:
            img = src.read([1,2,3]).transpose(1,2,0).astype(np.uint8)
            original_h, original_w = img.shape[:2]

        if self.is_test:
            # выполняем трансформации только по картинке
            if self.transforms:
                augmented = self.transforms(image=img)
                image = augmented['image']
            else:
                image = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
            return image, file_name, (original_h, original_w)

        # строим маску на основе аннотаций (если есть)
        annots = self.annot_mapping.get(file_name, [])
        mask = polygons_to_mask(original_h, original_w, annots, class_to_id)

        # аугментация применяем к image и mask одновременно
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        # mask — tensor HxW с целыми значениями 0..C
        return image, mask.long()
