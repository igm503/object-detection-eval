import os

from .yolov8 import Yolo
from .rtdetr import RTDETR
from .lwdetr import LWDETR
from .dfine import DFINE

model_registry = {
    "yolov8": Yolo,
    "rtdetr": RTDETR,
    "lwdetr": LWDETR,
    "dfine": DFINE,
}


def get_model(model_name: str, weights_path: str):
    if model_name not in model_registry:
        raise ValueError(f"Unsupported model: {model_name}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    return model_registry[model_name](str(weights_path))
