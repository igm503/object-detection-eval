import argparse
from pathlib import Path
from tqdm import tqdm

from dataset import CocoDataset, DataLoader
from eval import mean_average_precision
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate object detection models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["yolov8", "rtdetr", "lwdetr", "dfine"],
        help="Model to use",
    )
    parser.add_argument("--weights", type=Path, required=True, help="Path to model weights")
    parser.add_argument(
        "--images", type=Path, required=True, help="Path to directory containing images"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to COCO format annotation file",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu", "mps"],
        help="Device to run evaluation on (default: cpu)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.images.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images}")
    if not args.annotations.exists():
        raise FileNotFoundError(f"Annotation file not found: {args.annotations}")

    model = get_model(args.model, args.weights)

    dataset = CocoDataset(str(args.images), str(args.annotations))
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    image_metadata = []
    for image_tensors, image_data in tqdm(data_loader, desc="Evaluating"):
        inputs = image_tensors.to(args.device)
        detections = model(inputs)
        for img, dets in zip(image_data, detections):
            img.detections = dets
            img.prepare_detections()
            image_metadata.append(img)

    map_score = mean_average_precision(image_metadata)
    print(f"\nmAP: {map_score:.4f}")


if __name__ == "__main__":
    main()
