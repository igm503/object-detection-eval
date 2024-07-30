from dataclasses import dataclass
from time import time

import numpy as np
from tqdm import tqdm


@dataclass
class Detection:
    class_id: int
    bbox: list[float]
    conf: float
    matched: bool = False
    overlap: float | None = None

    def scale(self, size, original_size):
        x1, y1, x2, y2 = self.bbox
        width_ratio = size[0] / original_size[0]
        height_ratio = size[1] / original_size[1]
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

    def scale_detections(self):
        assert self.detections is not None, "detections is None"
        for detection in self.detections:
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


def mean_average_precision(
    images: list[Image],
    class_id: int | None = None,
):
    if class_id is None:
        class_ids = get_classes(images)
    else:
        class_ids = [class_id]
    num_gt = sum([len(img.ground_truths) for img in images])
    average_precisions = []
    reset_time = 0
    match_time = 0
    precision_time = 0
    for threshold in tqdm(np.arange(0.5, 1, 0.05)):
        t0 = time()
        for image in images:
            image.reset_matches()
        reset_time += time() - t0
        t0 = time()
        matched_dets = []
        for image in images:
            for class_id in class_ids:
                class_dets = [det for det in image.detections if det.class_id == class_id]
                class_gts = [gt for gt in image.ground_truths if gt.class_id == class_id]
                new_matches = match_dets(class_dets, class_gts, threshold)
                matched_dets.extend(new_matches)
        match_time += time() - t0
        t1 = time()
        average_precisions.append(average_precision(matched_dets, num_gt, threshold))
        precision_time += time() - t1
    return sum(average_precisions) / len(average_precisions)


def average_precision(detections: list[Detection], num_gt: int, overlap_threshold: float):
    detections = sorted(detections, key=lambda x: x.conf, reverse=True)
    true_positives = 0
    false_positives = 0
    false_negatives = num_gt
    recall = 0
    auc = 0
    start = 1.0
    for detection in detections:
        if detection.conf < start:
            new_recall = true_positives / (true_positives + false_negatives)
            if true_positives + false_positives == 0:
                new_precision = 1
            else:
                new_precision = true_positives / (true_positives + false_positives)
            auc += new_precision * (new_recall - recall)
            start = detection.conf
            recall = new_recall
        if detection.matched:
            assert detection.overlap is not None, "matched detection has None for overlap"
            if detection.overlap >= overlap_threshold:
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
    return detections


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
    if x2 < x1 or y2 < y1:
        return 0.0
    return area(x1, y1, x2, y2)


def area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)
