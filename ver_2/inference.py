# inference_fixed.py
import os
import json
import cv2
import numpy as np
from pathlib import Path
import rasterio
import torch
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import get_transforms, IMG_SIZE, CLASS_NAMES

# Параметры
PATH = "C:/Users/LeMeS/PycharmProjects/Competition"
EVAL_IMG_DIR = PATH + "/evaluation"
SAMPLE_SUB_PATH = PATH + "/sample_answare.json"
OUT_SUB_PATH = PATH + "/submission.json"
MODEL_PATH = PATH + "/runs/best_unet_resnet34_epoch29_iou0.0685.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(CLASS_NAMES) + 1  # фон + классы

CONF_THRESHOLD = 0.35
MIN_CONTOUR_AREA = 25  # в пикселях

def build_model(encoder_name="resnet34"):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES
    )
    return model

def mask_to_polygons(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue
        eps = 0.05 * cv2.arcLength(cnt, True)  # упрощение
        approx = cv2.approxPolyDP(cnt, eps, True)
        flat = [int(x) for p in approx.reshape(-1, 2).tolist() for x in p]
        if len(flat) >= 6:
            polys.append(flat)
    return polys

def run_inference(model_path=MODEL_PATH):
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_tf = get_transforms(IMG_SIZE, train=False)

    # load sample template
    with open(SAMPLE_SUB_PATH, 'r') as f:
        submission_template = json.load(f)

    predictions_by_file = {}

    img_paths = sorted([p for p in Path(EVAL_IMG_DIR).glob("*.tif")])
    for img_path in tqdm(img_paths, desc="Eval"):
        file_name = img_path.name
        with rasterio.open(str(img_path)) as src:
            img = src.read([1,2,3]).transpose(1,2,0).astype(np.float32) / 255.0
            orig_h, orig_w = img.shape[:2]

        # preprocess
        augmented = test_tf(image=(img*255).astype(np.uint8))  # аугментации ожидают uint8
        input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # C,H,W

        preds_for_image = []

        for cid in range(1, NUM_CLASSES):
            prob_map = probs[cid]
            prob_map_orig = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            bin_mask = (prob_map_orig > CONF_THRESHOLD).astype(np.uint8)

            polygons = mask_to_polygons(bin_mask)

            for poly in polygons:
                # confidence_score = среднее значение вероятности внутри полигона
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                pts = np.array(poly).reshape(-1,2)
                cv2.fillPoly(mask, [pts], 1)
                score = float(prob_map_orig[mask==1].mean())

                if score < CONF_THRESHOLD:
                    continue

                preds_for_image.append({
                    "class": CLASS_NAMES[cid-1],
                    "confidence_score": score,
                    "segmentation": poly
                })

        predictions_by_file[file_name] = preds_for_image

    # insert predictions into template
    for img_info in submission_template.get('images', []):
        fn = img_info['file_name']
        img_info['annotations'] = predictions_by_file.get(fn, [])

    # save submission
    with open(OUT_SUB_PATH, 'w') as f:
        json.dump(submission_template, f, indent=2)

    print(f"Saved submission to {OUT_SUB_PATH}")

if __name__ == "__main__":
    run_inference()
