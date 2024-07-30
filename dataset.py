import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image

from eval import GroundTruth


class CocoDataset(Dataset):
    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir

        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        self.image_to_annotations = {}
        self.image_to_filename = {}
        for img in self.coco_data["images"]:
            self.image_to_filename[img["id"]] = img["file_name"]
            self.image_to_annotations[img["id"]] = []

        for annotation in self.coco_data["annotations"]:
            self.image_to_annotations[annotation["image_id"]].append(annotation)

        self.ids = list(self.image_to_filename.keys())

        self.transform = Compose([Resize((640, 640)), ToTensor()])

    def __getitem__(self, index):
        img_id = self.ids[index]

        file_name = self.image_to_filename[img_id]
        img_path = os.path.join(self.root_dir, file_name)
        img = Image.open(img_path).convert("RGB")

        annotations = self.image_to_annotations[img_id]

        bboxes = []
        class_ids = []
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            bboxes.append([x, y, x + w, y + h])
            class_ids.append(annotation["category_id"])

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        class_ids = torch.as_tensor(class_ids, dtype=torch.int64)
        img_id = torch.tensor([img_id])

        target = [GroundTruth(class_id, bbox) for class_id, bbox in zip(class_ids, bboxes)]

        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.ids)
