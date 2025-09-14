import os
import glob
import json
import numpy as np
import pandas as pd
import rasterio
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ================= CONFIG ===================
class CFG:
    BASE_PATH = r"C:\Users\LeMeS\PycharmProjects\Competition"
    TRAIN_IMG_PATH = os.path.join(BASE_PATH, "train")
    EVAL_IMG_PATH = os.path.join(BASE_PATH, "evaluation")
    TRAIN_ANNOT_PATH = os.path.join(BASE_PATH, "train_annotations.json")
    SAVE_PATH = os.path.join(BASE_PATH, "runs_optune")
    SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_answare.json")

    IMG_SIZE = 1024
    BATCH_SIZE = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLASS_NAMES = ["individual_tree", "group_of_trees"]
    CONF_THRESHOLD = 0.5
    MIN_CONTOUR_AREA = 25

# ================= PARSE ANNOTATIONS ===================
def parse_annotations(annot_path):
    """Возвращает DataFrame с одной строкой на каждый инстанс"""
    with open(annot_path, 'r') as f:
        data = json.load(f)

    rows = []
    for img_info in data['images']:
        for annot in img_info.get('annotations', []):
            rows.append({
                'file_name': img_info['file_name'],
                'height': img_info['height'],
                'width': img_info['width'],
                'scene_type': img_info.get('scene_type', "unknown"),
                'cm_resolution': img_info.get('cm_resolution', 10),
                'class': annot['class'],
                'segmentation': annot['segmentation']
            })
    return pd.DataFrame(rows)

# ================= CLASS MAPPINGS ===================
class_to_id = {name: i + 1 for i, name in enumerate(CFG.CLASS_NAMES)}
id_to_class = {i + 1: name for i, name in enumerate(CFG.CLASS_NAMES)}
NUM_CLASSES = len(CFG.CLASS_NAMES) + 1  # +1 для background

# ================= DATASET ===================
class CanopyDataset(Dataset):
    def __init__(self, image_dir, df=None, class_to_id=None, transforms=None, is_test=False, return_filename=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        self.df = df
        self.class_to_id = class_to_id
        self.transforms = transforms
        self.is_test = is_test
        self.return_filename = return_filename

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        file_name = os.path.basename(image_path)

        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)

        if self.is_test:
            original_size = image.shape[:2]
            if self.transforms:
                image = self.transforms(image=image)['image']
            if self.return_filename:
                return image, file_name, original_size
            return image, original_size

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.int32)

        if self.df is not None:
            annots = self.df[self.df['file_name'] == file_name]
            for _, row in annots.iterrows():
                class_id = self.class_to_id.get(row['class'])
                if class_id:
                    seg = np.array(row['segmentation'], dtype=np.int32).reshape(-1, 2)
                    epsilon = max(1.0, 0.01 * cv2.arcLength(seg, True))
                    poly = cv2.approxPolyDP(seg, epsilon, True).reshape(-1, 2)
                    cv2.fillPoly(mask, [poly], color=class_id)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        if self.return_filename:
            return image, mask.long(), file_name
        return image, mask.long()
