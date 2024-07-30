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


@dataclass
class GroundTruth:
    class_id: int
    bbox: list[float]
    matched: bool = False


def reset(dets_or_gts: list[list[Detection | GroundTruth]]):
    for img in dets_or_gts:
        for det_or_gt in img:
            det_or_gt.matched = False


def get_classes(ground_truths: list[list[GroundTruth]]):
    classes = set()
    for img in ground_truths:
        for ground_truth in img:
            classes.add(ground_truth.class_id)
    return classes


def mean_average_precision(
    detections: list[list[Detection]],
    ground_truths: list[list[GroundTruth]],
    class_id: set[int] | None = None,
):
    if class_id is None:
        classes = get_classes(ground_truths)
    num_gt = sum([len(img) for img in ground_truths])
    average_precisions = []
    reset_time = 0
    match_time = 0
    precision_time = 0
    for threshold in tqdm(np.arange(0.5, 1, 0.05)):
        t0 = time()
        reset(detections)
        reset(ground_truths)
        reset_time += time() - t0
        t0 = time()
        matched_dets = []
        for img_dets, img_gts in zip(detections, ground_truths):
            for class_id in classes:
                class_dets = [det for det in img_dets if det.class_id == class_id]
                class_gts = [gt for gt in img_gts if gt.class_id == class_id]
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


ground_truths = [[GroundTruth(0, [i, i, i+10, i+10]) for i in range(100)]] * 10
detections = [[Detection(0, [i-1, i-1, i+10, i+10], i / 100) for i in range(100)]] * 10
print(mean_average_precision(detections, ground_truths))

