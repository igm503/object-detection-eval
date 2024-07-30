import os
import json
import torch
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import PILImage

from eval import GroundTruth, Image


class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            raise StopIteration
        batch_input = []
        batch_target = []
        for _ in range(self.batch_size):
            if self.idx >= len(self.dataset):
                break
            input, target = self.dataset[self.idx]
            batch_input.append(input)
            batch_target.append(target)
            self.idx += 1
        return torch.stack(batch_input), batch_target


class CocoDataset:
    SIZE = (640, 640)

    def __init__(self, root_dir, annotation_file):
        self.root_dir = root_dir

        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        self.image_to_annotations = {}
        self.images = {}
        for img in self.coco_data["images"]:
            self.images[img["id"]] = img
            self.image_to_annotations[img["id"]] = []

        for annotation in self.coco_data["annotations"]:
            self.image_to_annotations[annotation["image_id"]].append(annotation)

        self.ids = list(self.images.keys())

        self.transform = Compose([Resize(CocoDataset.SIZE), ToTensor()])

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_data = self.images[img_id]
        file_name = img_data["file_name"]
        img_path = os.path.join(self.root_dir, file_name)
        img = PILImage.open(img_path).convert("RGB")
        annotations = self.image_to_annotations[img_id]
        bboxes = []
        class_ids = []
        for annotation in annotations:
            x, y, w, h = annotation["bbox"]
            bboxes.append([x, y, x + w, y + h])
            class_ids.append(annotation["category_id"])
        target = Image(
            [GroundTruth(class_id, bbox) for class_id, bbox in zip(class_ids, bboxes)],
            (img_data["width"], img_data["height"]),
            CocoDataset.SIZE,
        )
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.ids)
