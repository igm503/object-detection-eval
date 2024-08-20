from dataclasses import dataclass

import numpy as np
from tqdm import tqdm


def validate_bbox(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    if not (-1 <= x1 < x2 <= image_width + 1 and -1 <= y1 < y2 <= image_height + 1):
        raise ValueError(f"Invalid bounding box: {bbox}")


def validate_confidence(conf):
    if not (0 <= conf <= 1):
        raise ValueError(f"Invalid confidence score: {conf}")


@dataclass
class Detection:
    class_id: int
    bbox: list[float]
    conf: float
    matched: bool = False
    overlap: float | None = None

    def __post_init__(self):
        validate_confidence(self.conf)

    def scale(self, new_size, original_size):
        x1, y1, x2, y2 = self.bbox
        width_ratio = new_size[0] / original_size[0]
        height_ratio = new_size[1] / original_size[1]
        self.bbox = [
            x1 * width_ratio,
            y1 * height_ratio,
            x2 * width_ratio,
            y2 * height_ratio,
        ]


@dataclass
class GroundTruth:
    class_id: int
    bbox: list[float]
    matched: bool = False


@dataclass
class Image:
    ground_truths: list[GroundTruth]
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    detections: list[Detection] | None = None

    def prepare_detections(self):
        assert self.detections is not None, "detections is None"
        for detection in self.detections:
            validate_bbox(detection.bbox, self.processed_size[0], self.processed_size[1])
            detection.scale(self.original_size, self.processed_size)

    def reset_matches(self):
        if self.detections is not None:
            for det in self.detections:
                det.matched = False
        for gt in self.ground_truths:
            gt.matched = False


def get_classes(images: list[Image]):
    classes = set()
    for img in images:
        for ground_truth in img.ground_truths:
            classes.add(ground_truth.class_id)
    return classes


def mean_average_precision(images: list[Image], class_id: int | None = None):
    average_precisions = []
    for threshold in tqdm(np.arange(0.5, 1, 0.05)):
        average_precisions.append(average_precision(images, threshold, class_id))
    return sum(average_precisions) / len(average_precisions)


def average_precision(images: list[Image], threshold: float, class_id: int | None = None):
    if class_id is None:
        class_ids = get_classes(images)
    else:
        class_ids = [class_id]
    num_gt = 0
    dets = []
    for image in images:
        image.reset_matches()
        for class_id in class_ids:
            class_dets = [det for det in image.detections if det.class_id == class_id]
            class_gts = [gt for gt in image.ground_truths if gt.class_id == class_id]
            match_dets(class_dets, class_gts, threshold)
            dets.extend(class_dets)
            num_gt += len(class_gts)
    return compute_average_precision(dets, num_gt)


def compute_average_precision(detections: list[Detection], num_gt: int):
    assert num_gt > 0
    if not detections:
        return 0
    detections = sorted(detections, key=lambda x: x.conf, reverse=True)
    true_positives = 0
    false_positives = 0
    false_negatives = num_gt
    recall = 0
    auc = 0
    start = detections[0].conf
    for detection in detections:
        if detection.conf < start:
            new_recall = true_positives / (true_positives + false_negatives)
            new_precision = true_positives / (true_positives + false_positives)
            auc += new_precision * (new_recall - recall)
            start = detection.conf
            recall = new_recall
        if detection.matched:
            true_positives += 1
            false_negatives -= 1
        else:
            false_positives += 1
    new_recall = true_positives / (true_positives + false_negatives)
    new_precision = true_positives / (true_positives + false_positives)
    auc += new_precision * (new_recall - recall)
    return auc


def match_dets(
    detections: list[Detection],
    ground_truths: list[GroundTruth],
    overlap_threshold: float,
):
    detections = sorted(detections, key=lambda x: x.conf, reverse=True)
    for detection in detections:
        candidate_iou = 0
        candidate_match = None
        for ground_truth in ground_truths:
            if ground_truth.matched:
                continue
            overlap = iou(detection.bbox, ground_truth.bbox)
            if overlap > candidate_iou and overlap > overlap_threshold:
                candidate_iou = overlap
                candidate_match = ground_truth
        if candidate_match:
            candidate_match.matched = True
            detection.matched = True
            detection.overlap = candidate_iou


def iou(bbox1, bbox2):
    inter = intersection(bbox1, bbox2)
    union = area(*bbox1) + area(*bbox2) - inter
    return inter / (union + 1e-6)


def intersection(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)
    return area(x1, y1, x2, y2)


def area(x1, y1, x2, y2):
    if x2 < x1 or y2 < y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)
