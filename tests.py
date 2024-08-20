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
    assert np.isclose(mAP, 1.0)
    gts = [[GroundTruth(0, [i, i, i + 10, i + 10]) for i in range(100)]] * 10
    dets = [[Detection(0, [i - 1, i - 1, i + 10, i + 10], i / 100) for i in range(100)]] * 10
    images = [Image(gts[i], (100, 100), (100, 100), dets[i]) for i in range(10)]
    mAP = mean_average_precision(images)
    assert abs(mAP - 0.693) < 0.001


def test_average_precision_calculation():
    dets = [
        Detection(1, [0, 0, 1, 1], 0.9, True, 0.9),
        Detection(1, [1, 1, 2, 2], 0.8, False),
    ]
    ap = compute_average_precision(dets, 2, 0.5)
    assert ap == 0.5


def test_match_dets():
    dets = [Detection(1, [0, 0, 1, 1], 0.9), Detection(1, [1, 1, 2, 2], 0.8)]
    gts = [GroundTruth(1, [0, 0, 1, 1])]
    matched = match_dets(dets, gts, 0.5)
    assert matched[0].matched
    assert not matched[1].matched


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
    assert intersection(bbox1, bbox2) == 1


def test_area():
    assert area(0, 0, 1.2, 2) == 2.4
