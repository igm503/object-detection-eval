from ultralytics import YOLO
from tqdm import tqdm
import torch
import onnxruntime as ort
import numpy as np

from dataset import CocoDataset, DataLoader
from eval import mean_average_precision, Detection

MEAN = (255 * np.array([0.485, 0.456, 0.406]))[:, None, None]
STD = (255 * np.array([0.229, 0.224, 0.225]))[:, None, None]


class RTDetr:
    def __init__(self, path: str):
        ort.set_default_logger_severity(3)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, sess_options, providers=providers)

    def __call__(self, images):
        images = images.cpu().numpy()
        orig_sizes = np.array([np.array([640, 640]) for _ in range(len(images))])
        images = images.astype(np.float32).transpose(0, 1, 2, 3)
        outputs = self.session.run(None, {"images": images, "orig_target_sizes": orig_sizes})
        batch_classes = outputs[0]
        batch_bboxes = outputs[1]
        batch_confs = outputs[2]
        detections = []
        for class_ids, bboxes, confs in zip(batch_classes, batch_bboxes, batch_confs):
            dets = [
                Detection(int(class_id), bbox, conf)
                for class_id, bbox, conf in zip(class_ids, bboxes, confs)
            ]
            detections.append(dets)
        return detections


class D_FINE:
    def __init__(self, path: str):
        ort.set_default_logger_severity(3)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, sess_options, providers=providers)

    def __call__(self, images):
        images = images.cpu().numpy()
        orig_sizes = np.array([np.array([640, 640]) for _ in range(len(images))])
        images = images.astype(np.float32).transpose(0, 1, 2, 3)
        outputs = self.session.run(None, {"images": images, "orig_target_sizes": orig_sizes})
        batch_classes = outputs[0]
        batch_bboxes = outputs[1]
        batch_confs = outputs[2]
        detections = []
        for class_ids, bboxes, confs in zip(batch_classes, batch_bboxes, batch_confs):
            dets = [
                Detection(int(class_id), bbox, conf)
                for class_id, bbox, conf in zip(class_ids, bboxes, confs)
            ]
            detections.append(dets)
        return detections


class LWDETR:
    def __init__(self, path):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, sess_options, providers=providers)

    def __call__(self, input):
        """
        Only supports single frame input
        """
        input = input.cpu().numpy()
        input = (input - MEAN) / STD
        input = input.astype(np.float32)
        output = self.session.run(None, {"input": input})

        logits, out_bbox = torch.Tensor(output[1]), torch.Tensor(output[0])

        prob = logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(logits.shape[0], -1), 300, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // logits.shape[2]
        labels = topk_indexes % logits.shape[2]
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        boxes[:, :, (0, 2)] *= input.shape[1]
        boxes[:, :, (1, 3)] *= input.shape[0]

        detections = []
        for bbox, class_id, score in zip(
            boxes[0].cpu().numpy(), labels[0].cpu().numpy(), scores[0].cpu().numpy()
        ):
            detections.append(Detection(class_id, bbox, score))

        return detections

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [
            (x_c - 0.5 * w.clamp(min=0.0)),
            (y_c - 0.5 * h.clamp(min=0.0)),
            (x_c + 0.5 * w.clamp(min=0.0)),
            (y_c + 0.5 * h.clamp(min=0.0)),
        ]
        return torch.stack(b, dim=-1)


class Yolo:
    def __init__(self, model: str):
        self.model = YOLO(model)

    def __call__(self, images):
        output = self.model(images, conf=0.0, device=0, half=False, imgsz=[640, 640], verbose=False)
        return [
            [
                Detection(int(bbox[5].item()), bbox[0:4].tolist(), bbox[4].item())
                for bbox in img.boxes.data.cpu()
            ]
            for img in output
        ]


model = Yolo("yolov8n.pt")

dataset = CocoDataset("path/to/images/dir/", "path/to/anno/file.json")
data_loader = DataLoader(dataset, batch_size=8)

image_metadata = []
for image_tensors, image_data in tqdm(data_loader):
    inputs = image_tensors.cuda()
    detections = model(inputs)
    for img, dets in zip(image_data, detections):
        img.detections = dets
        img.prepare_detections()
        image_metadata.append(img)

print(mean_average_precision(image_metadata))
