from pathlib import Path
import torch
from ultralytics import YOLO

# Классы
CLASS_NAMES = ["individual_tree", "group_of_trees"]
CLASS_WEIGHTS = [0.25, 0.75]  # редкий класс group_of_trees весом больше

# Цвета для визуализации масок
CLASS_COLORS = {
    0: (0, 255, 0),  # individual_tree — зеленый
    1: (0, 0, 255)  # group_of_trees — красный
}

# Проверяем GPU
device = 0 if torch.cuda.is_available() else "cpu"


def train_model(data_yaml: str, model_name: str = "yolov8s-seg.pt",
                epochs: int = 300, batch_size: int = 8, save_period: int = 5):
    torch.cuda.empty_cache()
    model = YOLO(model_name)

    # --- Аугментации ---
    augmentations = dict(
        mosaic=True,
        mixup=0.5,
        copy_paste=0.5,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.8,
        hsv_v=0.4,
        scale=0.6,
        translate=0.3,
        degrees=15
    )

    # --- Тренировка ---
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=736,
        batch=batch_size,
        device=device,
        pretrained=True,
        half=torch.cuda.is_available(),
        name=f"{Path(model_name).stem}_trees_aug",
        save_period=save_period,
        patience=50,
        **augmentations
    )



def inference_with_tta(model_path: str, source: str, conf: float = 0.25):
    """
    Инференс с TTA и низким порогом confidence для ловли редких объектов
    """
    model = YOLO(model_path)

    # TTA: зеркальное отражение + масштабирование
    results = model.predict(
        source=source,
        imgsz=736,
        conf=conf,
        augment=True,  # включает TTA
        half=torch.cuda.is_available()
    )
    return results


if __name__ == "__main__":
    dataset_dir = Path("C:/Users/LeMeS/PycharmProjects/Competition/dataset")
    data_yaml_path = dataset_dir / "data.yaml"

    # --- Обучение ---
    train_model(
        data_yaml=str(data_yaml_path),
        model_name="yolov8m-seg.pt",  # можно сменить на yolov8s-seg.pt / yolov8l-seg.pt
        epochs=200,
        batch_size=4,
        save_period=5
    )

    # --- Пример инференса с TTA ---
    # results = inference_with_tta("runs/segment/yolov8m-seg_trees_focal_aug/weights/best.pt",
    #                              "C:/Users/LeMeS/PycharmProjects/Competition/evaluation",
    #                              conf=0.25)
