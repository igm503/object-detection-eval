from ultralytics import YOLO

from dataset import CocoDataset, DataLoader
from eval import mean_average_precision, Detection


class Yolo:
    def __init__(self, model: str):
        self.model = YOLO(model)

    def __call__(self, images):
        output = self.model(images, conf=0, device=0, half=True, imgsz=[640, 640], verbose=False)
        return [
            [Detection(bbox[5].item(), bbox[0:4], bbox[4].item()) for bbox in img.boxes.data.cpu()]
            for img in output
        ]


model = Yolo("yolov8n.pt")

dataset = CocoDataset("path/to/images/dir/", "path/to/anno/file.json")
data_loader = DataLoader(dataset, batch_size=8)

image_metadata = []
for image_tensors, image_data in data_loader:
    inputs = image_tensors.cuda()
    detections = model(images)
    ground_truths.extend(gts)

print(mean_average_precision(detections, ground_truths))
