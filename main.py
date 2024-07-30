from ultralytics import YOLO
from torch.utils.data import DataLoader

from dataset import CocoDataset
from eval import mean_average_precision, Detection


class Yolo:
    def __init__(self, model: str):
        self.model = YOLO(model)

    def __call__(self, images):
        output = self.model(images, conf=0, device=0, half=True, imgsz=[640, 640], verbose=False)
        bboxes = output.boxes.data.cpu()
        return [Detection(bbox[4].item(), bbox[0:4], bbox[5].item()) for bbox in bboxes]


model = Yolo("yolov8n.pt")

dataset = CocoDataset("path/to/images/dir/", "path/to/anno/file.json")
data_loader = DataLoader(dataset, batch_size=8)

detections = []
ground_truths = []

for images, gts in data_loader:
    images = images.cuda()
    detections.append(model(images))
    ground_truths.append(gts)

print(mean_average_precision(detections, ground_truths))
