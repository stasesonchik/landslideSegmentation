from pathlib import Path
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = ["individual_tree", "group_of_trees"]

def visualize_segmentation(image_path: str, model_path: str, alpha: float = 0.5, conf: float = 0.3, tta: bool = True):
    """
    Визуализирует результаты сегментации модели YOLOv8 на одном изображении с TTA.

    :param image_path: путь к изображению
    :param model_path: путь к обученной модели .pt
    :param alpha: прозрачность маски
    :param conf: порог уверенности для предсказаний
    :param tta: включить Test-Time Augmentation
    """
    # Загружаем модель
    model = YOLO(model_path)

    # Прогоняем инференс с TTA
    results = model.predict(
        source=image_path,
        imgsz=1024,
        conf=conf,
        save=False,
        augment=tta  # включаем TTA
    )

    # Загружаем изображение
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Получаем маски
    result = results[0]
    masks = result.masks.xy if result.masks is not None else []

    # Рисуем маски
    overlay = img.copy()
    for mask_poly in masks:
        pts = np.array(mask_poly, np.int32).reshape(-1, 2)
        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Показываем результат
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def train_model(data_yaml: str, pretrained_model: str, epochs: int = 100, batch_size: int = 4, save_period: int = 5):
    """
    Обучает модель YOLOv8-seg на подготовленном датасете с аугментациями.

    :param data_yaml: путь к data.yaml
    :param pretrained_model: путь к .pt модели (yolov8s-seg.pt или предыдущие веса)
    :param epochs: количество эпох
    :param batch_size: размер батча
    :param save_period: сохранять модель каждые N эпох
    """
    torch.cuda.empty_cache()

    model = YOLO(pretrained_model)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=512,          # под GTX 1070 (8GB)
        batch=batch_size,
        device=0,           # GPU
        pretrained=True,
        name="yolov8s_seg_trees",
        half=True,          # FP16 для экономии памяти
        save_period=save_period,
        mosaic=True,
        mixup=False,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        translate=0.2,
        degrees=15
    )


if __name__ == "__main__":
    dataset_dir = Path("C:/Users/LeMeS/PycharmProjects/Competition/dataset")
    data_yaml_path = dataset_dir / "data.yaml"

    # --- Визуализация примера с TTA ---
    visualize_segmentation(
        image_path="C:/Users/LeMeS/PycharmProjects/Competition/evaluation/10cm_evaluation_1.tif",
        model_path="runs/segment/yolov8s_seg_trees4/weights/best.pt",
        tta=True
    )

    # --- Обучение модели с аугментациями и 100 эпох ---
    train_model(
        data_yaml=str(data_yaml_path),
        pretrained_model="yolov8s-seg.pt",
        epochs=100,
        batch_size=4,
        save_period=5
    )
