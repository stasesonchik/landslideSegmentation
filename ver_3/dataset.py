import os
import glob
import json
import numpy as np
import pandas as pd
import rasterio
import cv2
import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ================= CONFIG ===================
class CFG:
    # Paths
    BASE_PATH = r"C:\Users\LeMeS\PycharmProjects\Competition"
    TRAIN_IMG_PATH = os.path.join(BASE_PATH, "train")
    EVAL_IMG_PATH = os.path.join(BASE_PATH, "evaluation")
    TRAIN_ANNOT_PATH = os.path.join(BASE_PATH, "train_annotations.json")
    SAVE_PATH = os.path.join(BASE_PATH, "runs_optune")
    SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_answare.json")

    # Model
    MODEL_ARC = "Segformer"
    BACKBONE = "efficientnet-b3"
    ENCODER_WEIGHTS = "imagenet"

    # Training
    IMG_SIZE = 1024
    BATCH_SIZE = 2
    EPOCHS = 40
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classes
    CLASS_NAMES = ["individual_tree", "group_of_trees"]

    # Postprocessing
    CONF_THRESHOLD = 0.5
    MIN_CONTOUR_AREA = 25

# ================= PARSE ANNOTATIONS ===================
def parse_annotations(annot_path):
    with open(annot_path, 'r') as f:
        data = json.load(f)

    images_data = data['images']
    annotations_list = []

    for img_info in images_data:
        for annot in img_info.get('annotations', []):
            annotations_list.append({
                'file_name': img_info['file_name'],
                'height': img_info['height'],
                'width': img_info['width'],
                'class': annot['class'],
                'segmentation': annot['segmentation']
            })
    return pd.DataFrame(annotations_list)

# Class mappings
class_to_id = {name: i + 1 for i, name in enumerate(CFG.CLASS_NAMES)}
id_to_class = {i + 1: name for i, name in enumerate(CFG.CLASS_NAMES)}
NUM_CLASSES = len(CFG.CLASS_NAMES) + 1  # +1 for background

# ================= DATASET ===================
class CanopyDataset(Dataset):
    def __init__(self, image_dir, df, class_to_id, transforms=None, is_test=False):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        self.df = df
        self.class_to_id = class_to_id
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        file_name = os.path.basename(image_path)

        with rasterio.open(image_path) as src:
            image = src.read([1,2,3]).transpose(1,2,0).astype(np.uint8)

        if self.is_test:
            original_size = image.shape[:2]
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed['image']
            return image, file_name, original_size

        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.int32)

        annots = self.df[self.df['file_name'] == file_name]
        for _, row in annots.iterrows():
            class_id = self.class_to_id.get(row['class'])
            if class_id:
                segmentation = np.array(row['segmentation'], dtype=np.int32).reshape(-1, 2)
                epsilon = 5.0
                simplified = cv2.approxPolyDP(segmentation, epsilon, True)
                simplified = simplified.reshape(-1, 2)
                cv2.fillPoly(mask, [simplified], color=class_id)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask.long()
