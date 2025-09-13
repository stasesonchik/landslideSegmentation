import os
import json
import cv2
import numpy as np
import rasterio
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
from dataset import get_transforms, CFG

DEVICE=CFG.DEVICE
NUM_CLASSES=len(CFG.CLASS_NAMES)+1
IMG_SIZE=CFG.IMG_SIZE
CONF_THRESHOLD=0.35
MIN_CONTOUR_AREA=25

SCENE_WEIGHTS={
    "agriculture_plantation":2.0,
    "urban_area":1.5,
    "rural_area":1.0,
    "industrial_area":1.25,
    "open_field":1.0
}
RESOLUTION_WEIGHTS={"10":1.0,"20":1.25,"40":2.0,"60":2.5,"80":3.0}

# -------------------------
def build_model(encoder_name="resnet34"):
    model=smp.Unet(encoder_name=encoder_name,encoder_weights=None,in_channels=3,classes=NUM_CLASSES)
    return model.to(DEVICE)

# -------------------------
def mask_to_polygons(binary_mask):
    contours,_=cv2.findContours(binary_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    polys=[]
    for cnt in contours:
        if cv2.contourArea(cnt)<MIN_CONTOUR_AREA: continue
        eps=0.05*cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,eps,True)
        flat=[int(x) for p in approx.reshape(-1,2).tolist() for x in p]
        if len(flat)>=6: polys.append(flat)
    return polys

# -------------------------
def run_inference(model_path):
    model=build_model()
    model.load_state_dict(torch.load(model_path,map_location=DEVICE))
    model.eval()
    _,val_tf=get_transforms(IMG_SIZE)

    with open(CFG.SAMPLE_SUB_PATH,'r') as f:
        submission=json.load(f)

    img_paths=sorted([os.path.join(CFG.EVAL_IMG_PATH,p) for p in os.listdir(CFG.EVAL_IMG_PATH) if p.endswith(".tif")])
    for img_path in tqdm(img_paths):
        file_name=os.path.basename(img_path)
        with rasterio.open(img_path) as src:
            img=src.read([1,2,3]).transpose(1,2,0).astype(np.uint8)
            orig_h,orig_w=img.shape[:2]
            scene_type=src.tags().get("scene_type","rural_area")
            resolution=str(int(src.tags().get("cm_resolution",10)))

        augmented=val_tf(image=img)
        input_tensor=augmented['image'].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits=model(input_tensor)
            probs=torch.softmax(logits,dim=1).squeeze(0).cpu().numpy()

        preds_for_image=[]
        for cid in range(1,NUM_CLASSES):
            prob_map=cv2.resize(probs[cid],(orig_w,orig_h),interpolation=cv2.INTER_LINEAR)
            bin_mask=(prob_map>CONF_THRESHOLD).astype(np.uint8)
            polygons=mask_to_polygons(bin_mask)
            for poly in polygons:
                mask=np.zeros((orig_h,orig_w),dtype=np.uint8)
                pts=np.array(poly).reshape(-1,2)
                cv2.fillPoly(mask,[pts],1)
                score=float(prob_map[mask==1].mean())
                if score<CONF_THRESHOLD: continue
                weighted_score=score*SCENE_WEIGHTS.get(scene_type,1.0)*RESOLUTION_WEIGHTS.get(resolution,1.0)
                preds_for_image.append({"class":CFG.CLASS_NAMES[cid-1],
                                        "confidence_score":weighted_score,
                                        "segmentation":poly,
                                        "scene_type":scene_type,
                                        "resolution":resolution})
        # вставляем в submission
        for img_info in submission.get("images",[]):
            if img_info["file_name"]==file_name:
                img_info["annotations"]=preds_for_image

    # сохраняем сабмит
    out_path=os.path.join(CFG.BASE_PATH,"submission.json")
    with open(out_path,'w') as f:
        json.dump(submission,f,indent=2)
    print(f"Saved submission: {out_path}")

# -------------------------
if __name__=="__main__":
    MODEL_PATH=os.path.join(CFG.SAVE_PATH,"best_model.pth")
    run_inference(MODEL_PATH)
