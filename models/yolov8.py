from ultralytics import YOLO
from ..eval import Detection


class Yolo:
    def __init__(self, model: str, device: str):
        self.model = YOLO(model)
        if device == "cuda":
            self.device = 0
        else:
            self.device = device

    def __call__(self, images):
        output = self.model(
            images,
            conf=0.0,
            device=self.device,
            half=False,
            imgsz=[640, 640],
            verbose=False,
        )
        return [
            [
                Detection(int(bbox[5].item()), bbox[0:4].tolist(), bbox[4].item())
                for bbox in img.boxes.data.cpu()
            ]
            for img in output
        ]
