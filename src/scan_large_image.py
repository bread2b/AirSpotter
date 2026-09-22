from pathlib import Path
import argparse
import csv

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torch import nn
from torchvision import models, transforms


def pad_tile(tile, tile_width, tile_height):
    """Pad an edge tile by repeating its last row and column."""
    if tile.size == (tile_width, tile_height):
        return tile

    padded = Image.new("RGB", (tile_width, tile_height))
    padded.paste(tile, (0, 0))
    width, height = tile.size

    if width < tile_width:
        right_edge = tile.crop((width - 1, 0, width, height))
        right_edge = right_edge.resize((tile_width - width, height))
        padded.paste(right_edge, (width, 0))

    if height < tile_height:
        bottom_edge = padded.crop((0, height - 1, tile_width, height))
        bottom_edge = bottom_edge.resize((tile_width, tile_height - height))
        padded.paste(bottom_edge, (0, height))

    return padded


def make_group_tiles(
    image,
    columns,
    rows,
    tile_width,
    tile_height,
    group_size,
    offset_x=0,
    offset_y=0,
):
    """Build aligned, non-overlapping groups of base grid cells."""
    for row in range(0, rows, group_size):
        top = offset_y + row * tile_height
        for column in range(0, columns, group_size):
            left = offset_x + column * tile_width
            right = min(left + tile_width * group_size, image.width)
            bottom = min(top + tile_height * group_size, image.height)
            box = (left, top, right, bottom)
            tile = pad_tile(
                image.crop(box),
                tile_width * group_size,
                tile_height * group_size,
            )
            yield box, row, column, tile


def score_tiles(records, model, transform, airport_index, device, batch_size):
    scored = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch_records = records[start:start + batch_size]
            batch = torch.stack([
                transform(tile) for _, _, _, tile in batch_records
            ]).to(device)
            scores = torch.softmax(model(batch), dim=1)[:, airport_index]

            for (box, row, column, _), score in zip(
                batch_records,
                scores.cpu().tolist(),
            ):
                scored.append({
                    "box": box,
                    "row": row,
                    "column": column,
                    "airport_score": score,
                })
    return scored


def scan_offset(
    image,
    columns,
    rows,
    tile_width,
    tile_height,
    offset_x,
    offset_y,
    offset_name,
    model,
    transform,
    airport_index,
    device,
    batch_size,
    small_threshold,
    medium_threshold,
    large_threshold,
    combined_threshold,
    minimum_cells,
):
    cells_by_position = {}
    level_counts = {}

    for group_size in (1, 3, 4):
        records = list(make_group_tiles(
            image,
            columns,
            rows,
            tile_width,
            tile_height,
            group_size,
            offset_x,
            offset_y,
        ))
        level_counts[group_size] = len(records)
        scored_groups = score_tiles(
            records,
            model,
            transform,
            airport_index,
            device,
            batch_size,
        )

        for group in scored_groups:
            for row in range(group["row"], group["row"] + group_size):
                for column in range(
                    group["column"],
                    group["column"] + group_size,
                ):
                    position = (row, column)
                    if position not in cells_by_position:
                        left = offset_x + column * tile_width
                        top = offset_y + row * tile_height
                        cells_by_position[position] = {
                            "box": (
                                left,
                                top,
                                min(left + tile_width, image.width),
                                min(top + tile_height, image.height),
                            ),
                            "row": row,
                            "column": column,
                            "offset": offset_name,
                        }
                    cells_by_position[position][
                        f"score_{group_size}x{group_size}"
                    ] = group["airport_score"]

    cells = []
    selected = []
    for position in sorted(cells_by_position):
        cell = cells_by_position[position]
        small = cell["score_1x1"]
        medium = cell["score_3x3"]
        large = cell["score_4x4"]
        cell["airport_score"] = (
            0.20 * small + 0.35 * medium + 0.45 * large
        )
        cell["selected"] = (
            small >= small_threshold
            and (medium >= medium_threshold or large >= large_threshold)
            and cell["airport_score"] >= combined_threshold
        )
        cells.append(cell)
        if cell["selected"]:
            selected.append(cell)

    groups = merge_adjacent_tiles(selected, minimum_cells)
    for group in groups:
        group["offset"] = offset_name
    return cells, groups, level_counts


def overlap_fraction(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area else 0.0


def combine_offset_groups(groups, minimum_offset_hits=2):
    clusters = []
    for group in sorted(groups, key=lambda item: item["cell_count"], reverse=True):
        matching = next(
            (
                cluster for cluster in clusters
                if overlap_fraction(group["box"], cluster["box"]) >= 0.25
            ),
            None,
        )
        if matching is None:
            clusters.append({
                "box": group["box"],
                "offsets": {group["offset"]},
                "cell_count": group["cell_count"],
                "grid_score": group["average_score"],
            })
        else:
            matching["box"] = (
                min(matching["box"][0], group["box"][0]),
                min(matching["box"][1], group["box"][1]),
                max(matching["box"][2], group["box"][2]),
                max(matching["box"][3], group["box"][3]),
            )
            matching["offsets"].add(group["offset"])
            matching["cell_count"] += group["cell_count"]
            matching["grid_score"] = max(
                matching["grid_score"], group["average_score"]
            )

    return [
        cluster for cluster in clusters
        if len(cluster["offsets"]) >= minimum_offset_hits
    ]


def confirm_candidate(image, box, model, transform, airport_index, device):
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    margin_x = round(width * 0.25)
    margin_y = round(height * 0.25)
    expanded = (
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(image.width, right + margin_x),
        min(image.height, bottom + margin_y),
    )
    crop = image.crop(expanded)
    variants = [crop, ImageOps.mirror(crop), ImageOps.flip(crop)]
    batch = torch.stack([transform(item) for item in variants]).to(device)
    with torch.inference_mode():
        scores = torch.softmax(model(batch), dim=1)[:, airport_index]
    return expanded, sum(scores.cpu().tolist()) / len(variants)


def merge_adjacent_tiles(detections, minimum_cells=2):
    """Merge touching positive grid cells into larger candidate regions."""
    by_position = {
        (item["row"], item["column"]): item for item in detections
    }
    remaining = set(by_position)
    groups = []

    while remaining:
        start = remaining.pop()
        stack = [start]
        positions_in_group = [start]

        while stack:
            row, column = stack.pop()
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    neighbor = (
                        row + row_offset,
                        column + column_offset,
                    )
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
                        positions_in_group.append(neighbor)

        if len(positions_in_group) < minimum_cells:
            continue

        cells = [by_position[position] for position in positions_in_group]
        groups.append({
            "box": (
                min(item["box"][0] for item in cells),
                min(item["box"][1] for item in cells),
                max(item["box"][2] for item in cells),
                max(item["box"][3] for item in cells),
            ),
            "cell_count": len(cells),
            "max_score": max(item["airport_score"] for item in cells),
            "average_score": sum(
                item["airport_score"] for item in cells
            ) / len(cells),
        })

    return sorted(
        groups,
        key=lambda item: (item["cell_count"], item["max_score"]),
        reverse=True,
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(checkpoint["class_to_idx"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(tuple(checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["mean"],
            std=checkpoint["std"],
        ),
    ])
    return model, transform, checkpoint["class_to_idx"]["airport"]


def save_results(
    image,
    cells,
    candidates,
    image_path,
    output_dir,
    tile_width,
    tile_height,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output = output_dir / f"{image_path.stem}_detected.jpg"
    csv_output = output_dir / f"{image_path.stem}_detections.csv"
    tile_csv_output = output_dir / f"{image_path.stem}_all_scores.csv"

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for left in range(tile_width, image.width, tile_width):
        draw.line((left, 0, left, image.height), fill=(210, 210, 0), width=1)
    for top in range(tile_height, image.height, tile_height):
        draw.line((0, top, image.width, top), fill=(210, 210, 0), width=1)

    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default()

    for number, candidate in enumerate(candidates, start=1):
        box = candidate["box"]
        draw.rectangle(box, outline="red", width=8)
        label = (
            f"Candidate {number}: confirm {candidate['confirmation_score']:.1%}, "
            f"offsets {candidate['offset_hits']}"
        )
        text_box = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(text_box, fill="red")
        draw.text((box[0], box[1]), label, fill="white", font=font)

    annotated.save(image_output, quality=95)

    with csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "candidate",
            "left",
            "top",
            "right",
            "bottom",
            "cell_count",
            "offset_hits",
            "grid_score",
            "confirmation_score",
        ])
        for number, candidate in enumerate(candidates, start=1):
            writer.writerow([
                number,
                *candidate["box"],
                candidate["cell_count"],
                candidate["offset_hits"],
                candidate["grid_score"],
                candidate["confirmation_score"],
            ])

    with tile_csv_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "row",
            "column",
            "offset",
            "left",
            "top",
            "right",
            "bottom",
            "score_1x1",
            "score_3x3",
            "score_4x4",
            "combined_score",
            "selected",
        ])
        for cell in cells:
            writer.writerow([
                cell["row"],
                cell["column"],
                cell["offset"],
                *cell["box"],
                cell["score_1x1"],
                cell["score_3x3"],
                cell["score_4x4"],
                cell["airport_score"],
                cell["selected"],
            ])

    return image_output, csv_output, tile_csv_output


def main():
    parser = argparse.ArgumentParser(
        description="Find possible airport regions in a large image."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--grid-columns",
        type=int,
        default=48,
        help="Number of non-overlapping grid columns (default: 48).",
    )
    parser.add_argument(
        "--grid-rows",
        type=int,
        default=24,
        help="Number of non-overlapping grid rows (default: 24).",
    )
    parser.add_argument(
        "--small-threshold", type=float, default=0.65
    )
    parser.add_argument(
        "--medium-threshold", type=float, default=0.85
    )
    parser.add_argument(
        "--large-threshold", type=float, default=0.90
    )
    parser.add_argument(
        "--combined-threshold", type=float, default=0.78
    )
    parser.add_argument(
        "--confirmation-threshold", type=float, default=0.85
    )
    parser.add_argument(
        "--minimum-offset-hits", type=int, default=2
    )
    parser.add_argument(
        "--minimum-cells",
        type=int,
        default=2,
        help="Minimum connected positive cells per offset (default: 2).",
    )
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.grid_columns <= 0 or args.grid_rows <= 0:
        raise ValueError("Grid columns and rows must be positive.")
    if args.grid_columns % 12 or args.grid_rows % 12:
        raise ValueError(
            "Grid columns and rows must be divisible by both 3 and 4."
        )
    thresholds = (
        args.small_threshold,
        args.medium_threshold,
        args.large_threshold,
        args.combined_threshold,
        args.confirmation_threshold,
    )
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("All thresholds must be between 0 and 1.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.minimum_cells <= 0:
        raise ValueError("--minimum-cells must be positive.")
    if not 1 <= args.minimum_offset_hits <= 4:
        raise ValueError("--minimum-offset-hits must be between 1 and 4.")
    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, airport_index = load_model(args.checkpoint, device)

    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    tile_width = (image.width + args.grid_columns - 1) // args.grid_columns
    tile_height = (image.height + args.grid_rows - 1) // args.grid_rows
    offsets = [
        (0, 0, "original"),
        (tile_width // 2, 0, "half_x"),
        (0, tile_height // 2, "half_y"),
        (tile_width // 2, tile_height // 2, "half_xy"),
    ]
    all_cells = []
    all_groups = []
    total_checks = 0

    for offset_x, offset_y, offset_name in offsets:
        cells, groups, level_counts = scan_offset(
            image,
            args.grid_columns,
            args.grid_rows,
            tile_width,
            tile_height,
            offset_x,
            offset_y,
            offset_name,
            model,
            transform,
            airport_index,
            device,
            args.batch_size,
            args.small_threshold,
            args.medium_threshold,
            args.large_threshold,
            args.combined_threshold,
            args.minimum_cells,
        )
        all_cells.extend(cells)
        all_groups.extend(groups)
        total_checks += sum(level_counts.values())

    clusters = combine_offset_groups(
        all_groups,
        args.minimum_offset_hits,
    )
    candidates = []
    for cluster in clusters:
        confirmed_box, confirmation_score = confirm_candidate(
            image,
            cluster["box"],
            model,
            transform,
            airport_index,
            device,
        )
        if confirmation_score >= args.confirmation_threshold:
            cluster["box"] = confirmed_box
            cluster["confirmation_score"] = confirmation_score
            cluster["offset_hits"] = len(cluster["offsets"])
            candidates.append(cluster)

    candidates.sort(
        key=lambda item: (
            item["offset_hits"],
            item["confirmation_score"],
            item["cell_count"],
        ),
        reverse=True,
    )
    candidates = candidates[:args.max_candidates]
    image_output, csv_output, tile_csv_output = save_results(
        image,
        all_cells,
        candidates,
        args.image,
        args.output_dir,
        tile_width,
        tile_height,
    )

    print("Image:", args.image.resolve())
    print("Image size:", image.size)
    print("Device:", device)
    print("Grid:", f"{args.grid_columns} columns x {args.grid_rows} rows")
    print("Cell size:", f"{tile_width} x {tile_height}")
    print("Offset passes:", len(offsets))
    print("Model checks:", total_checks)
    print("Provisional regions:", len(all_groups))
    print("Cross-offset regions:", len(clusters))
    print("Final candidates:", len(candidates))
    print("Annotated image:", image_output.resolve())
    print("Detection CSV:", csv_output.resolve())
    print("All score CSV:", tile_csv_output.resolve())
    print("\nBoxes are approximate candidate regions, not exact boundaries.")


if __name__ == "__main__":
    main()
