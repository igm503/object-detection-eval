import numpy as np

from eval import (
    Detection,
    GroundTruth,
    Image,
    get_classes,
    mean_average_precision,
    compute_average_precision,
    match_dets,
    iou,
    intersection,
    area,
)


def test_reset():
    dets = [Detection(1, [0, 0, 1, 1], 0.9, True)]
    gts = [GroundTruth(1, [0, 0, 1, 1], True)] * 6
    image = Image(gts, (100, 100), (100, 100), dets)
    image.reset_matches()
    assert not dets[0].matched
    assert not gts[0].matched


def test_get_classes():
    gts = [
        [GroundTruth(1, [0, 0, 1, 1]), GroundTruth(2, [1, 1, 2, 2])],
        [GroundTruth(1, [2, 2, 3, 3]), GroundTruth(3, [3, 3, 4, 4])],
    ]
    image1 = Image(gts[0], (100, 100), (100, 100))
    image2 = Image(gts[1], (100, 100), (100, 100))
    images = [image1, image2]
    assert get_classes(images) == {1, 2, 3}


def test_mean_average_precision():
    dets = [Detection(1, [0, 0, 1, 1], 0.9)]
    gts = [GroundTruth(1, [0, 0, 1, 1])]
    image = Image(gts, (100, 100), (100, 100), dets)
    mAP = mean_average_precision([image])
    dets = [Detection(1, [0, 0, 1, 1], 0.9), Detection(0, [1, 1, 2, 2], 0.8)]
    gts = [GroundTruth(1, [0, 0, 1, 1]), GroundTruth(0, [1, 1, 2, 2])]
    image = Image(gts, (100, 100), (100, 100), dets)
    mAP = mean_average_precision([image], class_id=0)
    assert np.isclose(mAP, 1.0)
    gts = [[GroundTruth(0, [i, i, i + 10, i + 10]) for i in range(100)]] * 10
    dets = [[Detection(0, [i - 1, i - 1, i + 10, i + 10], i / 100) for i in range(100)]] * 10
    images = [Image(gts[i], (100, 100), (100, 100), dets[i]) for i in range(10)]
    mAP = mean_average_precision(images)
    assert np.isclose(mAP, 0.693)
    gts = [[GroundTruth(0, [i, i, i + 10, i + 10]) for i in range(100)]] * 10
    dets = [[Detection(0, [i - 1, i - 1, i + 10, i + 10], i / 100) for i in range(100)]] * 10
    images = [Image(gts[i], (100, 100), (100, 100), dets[i]) for i in range(10)]
    mAP = mean_average_precision(images)
    assert np.isclose(mAP, 0.693)

def test_average_precision_calculation():
    dets = [
        Detection(1, [0, 0, 1, 1], 0.9, True, 0.9),
        Detection(1, [1, 1, 2, 2], 0.8, False),
    ]
    ap = compute_average_precision(dets, 2)
    assert np.isclose(ap, 0.5)
    dets = [
        Detection(1, [0, 0, 1, 1], 0.9, False),
        Detection(1, [1, 1, 2, 2], 0.8, False),
    ]
    ap = compute_average_precision(dets, 2)
    assert np.isclose(ap, 0.0)


def test_match_dets():
    dets = [Detection(1, [0, 0, 1, 1], 0.9), Detection(1, [1, 1, 2, 2], 0.8)]
    gts = [GroundTruth(1, [0, 0, 1, 1])]
    match_dets(dets, gts, 0.5)
    assert dets[0].matched
    assert not dets[1].matched


def test_iou():
    bbox1 = [0, 0, 2, 2]
    bbox2 = [1, 1, 3, 3]
    assert np.isclose(iou(bbox1, bbox2), 1 / 7)
    bbox1 = [76, 126, 126, 176]
    bbox2 = [0, 0, 50, 42]
    assert np.isclose(iou(bbox1, bbox2), 0.0)
    bbox1 = [76, 126, 126, 176]
    bbox2 = [26, 126, 126, 176]
    assert np.isclose(iou(bbox1, bbox2), 0.5)


def test_intersection():
    bbox1 = [0, 0, 2, 2]
    bbox2 = [1, 1, 3, 3]
    assert np.isclose(intersection(bbox1, bbox2), 1)


def test_area():
    assert np.isclose(area(0, 0, 1.2, 2), 2.4)
