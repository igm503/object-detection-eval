# object-detection-eval
a short exercise in implementing mean average precision 0.5:0.95 calculation for object detection

## Warning

This toolkit does not calculate mAP in the same way as [pycocotools](https://github.com/ppwwyyxx/cocoapi). Results shouldn't be compared to mAP and AP values found in the literature.

## Installation

```
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```
python main.py \
    --model yolov8 \
    --weights path/to/weights.pt \
    --images path/to/image/dir \
    --annotations path/to/annotations.json \
    --batch-size 8 \
    --device cuda
```

Arguments:
- `--model`: Model architecture to use (choices: [yolov8](https://github.com/ultralytics/ultralytics), [rtdetr](https://github.com/lyuwenyu/RT-DETR/tree/main), [lwdetr](https://github.com/Atten4Vis/LW-DETR), [dfine](https://github.com/Peterande/D-FINE))
- `--weights`: Path to model weights
- `--images`: Directory containing eval images
- `--annotations`: Path to annotation file ([coco format](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html))
- `--batch-size`: default: 8
- `--device`: choices: cuda, cpu, mps, default: cpu

## Adding New Models

To add support for a new model architecture:

1. Create a new file in the `models` directory (e.g., `models/custom_model.py`)
2. Define your model class with the following interface:

```python
class CustomModel:
    def __init__(self, path: str, device: str):
        """
        Args:
            path: Path to model weights
            device: Device to run model on
        """
        pass

    def __call__(self, images): -> list[list[Detection]]
        """
        Run inference on a batch of images
        Args:
            images: Tensor of shape (batch_size, channels, height, width)
        Returns:
            list[list[Detection]]: list of detections for each image in the batch. 
        """

        return detections
```

3. Add your model to the registry in `models/__init__.py`:

```python
from .custom_model import CustomModel

model_registry = {
    "yolov8": Yolo,
    "rtdetr": RTDETR,
    "lwdetr": LWDETR,
    "dfine": DFINE,
    "custom": CustomModel,  # Add your model here
}
```

### Detection Class Interface

Your model's `__call__` method should return a list of detections for each image in the input batch, where each detection is an instance of:

```python
@dataclass
class Detection:
    class_id: int        # Class ID of the detection
    bbox: list[float]    # Bounding box coordinates [x1, y1, x2, y2]
    conf: float          # Confidence score
```

## Notes

- Input images will be tensors of shape (batch_size, channels, 640, 640). No transformations will have been applied other than resizing to (640, 640).
- All bounding box coordinates should be in absolute pixel coordinates [x1, y1, x2, y2]
- The evaluation framework handles scaling between processed and original image sizes. Your model just needs to return coordinates in the scale of the input tensors.

## License

The license for this project is MIT. However, one of the example models, [yolov8](https://github.com/ultralytics/ultralytics), requires weights and a library that are not so permissively licensed.
