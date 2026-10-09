"""Test a DIOR-trained YOLOv5 model on large satellite maps."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect airports in large satellite maps")
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        default=[
            ROOT / "data/manual/large_map.jpg",
            ROOT / "data/manual/large_map2.jpg",
            ROOT / "data/manual/large_map3.jpg",
        ],
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "models/pretrained/yolov5_dior_airport.pt",
    )
    parser.add_argument("--yolov5-dir", type=Path, default=ROOT / "third_party/yolov5")
    parser.add_argument("--output", type=Path, default=ROOT / "results/dior_airport")
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="", help="Empty selects CUDA when available")
    return parser.parse_args()


def starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    step = max(1, int(tile_size * (1.0 - overlap)))
    values = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if values[-1] != last:
        values.append(last)
    return values


def load_model(args: argparse.Namespace):
    sys.path.insert(0, str(args.yolov5_dir.resolve()))
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device

    device = select_device(args.device)
    model = DetectMultiBackend(
        str(args.weights.resolve()),
        device=device,
        fp16=device.type != "cpu",
    )
    airport_ids = [index for index, name in model.names.items() if name.lower() == "airport"]
    if not airport_ids:
        raise RuntimeError(f"The model has no airport class: {model.names}")
    return model, device, airport_ids


def prepare_batch(tiles: list[np.ndarray], image_size: int, stride: int) -> torch.Tensor:
    from utils.augmentations import letterbox

    prepared = []
    for tile in tiles:
        resized = letterbox(tile, image_size, stride=stride, auto=False)[0]
        resized = resized[:, :, ::-1].transpose(2, 0, 1)
        prepared.append(np.ascontiguousarray(resized))
    return torch.from_numpy(np.stack(prepared))


def scan_image(image_path: Path, model, device, airport_ids: list[int], args: argparse.Namespace):
    from utils.general import non_max_suppression, scale_boxes

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    height, width = image.shape[:2]
    tile_size = min(args.tile_size, width, height)
    positions = [
        (x, y)
        for y in starts(height, tile_size, args.overlap)
        for x in starts(width, tile_size, args.overlap)
    ]
    detections: list[list[float]] = []

    print(f"{image_path.name}: {width}x{height}, {len(positions)} tiles")
    for offset in range(0, len(positions), args.batch_size):
        batch_positions = positions[offset : offset + args.batch_size]
        tiles = [image[y : y + tile_size, x : x + tile_size] for x, y in batch_positions]
        tensor = prepare_batch(tiles, tile_size, int(model.stride)).to(device)
        tensor = tensor.half() if model.fp16 else tensor.float()
        tensor /= 255.0

        predictions = model(tensor)
        predictions = non_max_suppression(
            predictions,
            args.confidence,
            args.iou,
            classes=airport_ids,
            max_det=50,
        )

        for prediction, tile, (tile_x, tile_y) in zip(predictions, tiles, batch_positions):
            if not len(prediction):
                continue
            prediction[:, :4] = scale_boxes(tensor.shape[2:], prediction[:, :4], tile.shape).round()
            for x1, y1, x2, y2, confidence, class_id in prediction.cpu().tolist():
                detections.append(
                    [x1 + tile_x, y1 + tile_y, x2 + tile_x, y2 + tile_y, confidence, class_id]
                )

        done = min(offset + args.batch_size, len(positions))
        print(f"  tiles {done}/{len(positions)}", end="\r", flush=True)
    print()

    if detections:
        values = torch.tensor(detections, dtype=torch.float32)
        keep = nms(values[:, :4], values[:, 4], args.iou)
        detections = values[keep].tolist()

    return image, merge_overlapping(detections)


def merge_overlapping(detections: list[list[float]], minimum_overlap: float = 0.05):
    """Merge tile detections that cover different portions of the same airport."""
    groups: list[dict] = []
    for detection in sorted(detections, key=lambda item: item[4], reverse=True):
        x1, y1, x2, y2, confidence, class_id = detection
        matching = []
        for index, group in enumerate(groups):
            gx1, gy1, gx2, gy2 = group["box"]
            intersection = max(0.0, min(x2, gx2) - max(x1, gx1)) * max(
                0.0, min(y2, gy2) - max(y1, gy1)
            )
            smaller_area = min((x2 - x1) * (y2 - y1), (gx2 - gx1) * (gy2 - gy1))
            if smaller_area > 0 and intersection / smaller_area >= minimum_overlap:
                matching.append(index)

        if not matching:
            groups.append(
                {"box": [x1, y1, x2, y2], "confidence": confidence, "class_id": class_id, "support": 1}
            )
            continue

        base = groups[matching[0]]
        boxes = [base["box"], [x1, y1, x2, y2]]
        for index in reversed(matching[1:]):
            other = groups.pop(index)
            boxes.append(other["box"])
            base["confidence"] = max(base["confidence"], other["confidence"])
            base["support"] += other["support"]
        base["box"] = [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
        base["confidence"] = max(base["confidence"], confidence)
        base["support"] += 1

    return [
        [*group["box"], group["confidence"], group["class_id"], group["support"]]
        for group in groups
    ]


def save_results(image_path: Path, image: np.ndarray, detections: list[list[float]], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()
    csv_path = output / f"{image_path.stem}_detections.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x1", "y1", "x2", "y2", "confidence", "class", "supporting_tiles"])
        for x1, y1, x2, y2, confidence, _, support in detections:
            coords = [int(x1), int(y1), int(x2), int(y2)]
            writer.writerow([*coords, f"{confidence:.6f}", "airport", int(support)])
            cv2.rectangle(annotated, coords[:2], coords[2:], (0, 0, 255), 6)
            label = f"airport {confidence:.2f} ({int(support)} tiles)"
            cv2.putText(
                annotated,
                label,
                (coords[0], max(35, coords[1] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

    image_output = output / f"{image_path.stem}_airport.jpg"
    cv2.imwrite(str(image_output), annotated, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return image_output, csv_path


def main() -> None:
    args = parse_args()
    if not 0 <= args.overlap < 1:
        raise ValueError("--overlap must be between 0 and 1")

    model, device, airport_ids = load_model(args)
    print(f"Device: {device}; airport class: {airport_ids}")

    for image_path in args.images:
        image_path = image_path.resolve()
        image, detections = scan_image(image_path, model, device, airport_ids, args)
        image_output, csv_output = save_results(image_path, image, detections, args.output)
        print(f"  detections: {len(detections)}")
        print(f"  image: {image_output}")
        print(f"  csv:   {csv_output}")


if __name__ == "__main__":
    main()
