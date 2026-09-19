from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = DATA / "split"

# Train and validation ratios; the remainder is used for testing.
RATIOS = (0.70, 0.15)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Use a fixed seed for reproducible splits.
rng = random.Random(42)


def get_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def split_images(images):
    images = images.copy()
    rng.shuffle(images)

    train_end = round(len(images) * RATIOS[0])
    val_end = train_end + round(len(images) * RATIOS[1])

    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


# Validate the input data before copying any images.
airports = get_images(DATA / "airport")
negatives = get_images(DATA / "non_airport")

if len(airports) != 700 or len(negatives) != 700:
    raise ValueError(
        "Expected 700 airport and 700 non-airport images."
    )

# Group negative images by their original scene category.
# This ensures every category appears in all three splits.
negative_groups = {}

for path in negatives:
    # Example: railway_station_001.jpg -> railway_station
    category = path.stem.rsplit("_", 1)[0]
    negative_groups.setdefault(category, []).append(path)

if len(negative_groups) != 10 or any(
    len(group) != 70 for group in negative_groups.values()
):
    raise ValueError(
        "Expected 10 non-airport categories, 70 images each."
    )

# Prevent accidentally overwriting an existing split.
if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Move it elsewhere before rerunning."
    )

airport_splits = split_images(airports)
negative_splits = {"train": [], "val": [], "test": []}

for index, category in enumerate(sorted(negative_groups)):
    images = negative_groups[category].copy()
    rng.shuffle(images)

    # Assign 49 training images per category.
    # Alternate validation/test counts between 10/11 and 11/10.
    # Across 10 categories: 490 train, 105 validation, 105 test.
    val_count = 10 if index % 2 == 0 else 11

    negative_splits["train"].extend(images[:49])
    negative_splits["val"].extend(images[49:49 + val_count])
    negative_splits["test"].extend(images[49 + val_count:])

# Copy images into split folders while preserving the source files.
for split_name in ("train", "val", "test"):
    for label, splits in (
        ("airport", airport_splits),
        ("non_airport", negative_splits),
    ):
        destination = OUTPUT / split_name / label
        destination.mkdir(parents=True, exist_ok=True)

        for source in splits[split_name]:
            shutil.copy2(source, destination / source.name)

        print(f"{split_name}/{label}: {len(splits[split_name])}")

print("\nDataset split finished.")