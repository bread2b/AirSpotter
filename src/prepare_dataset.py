from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "RESISC45"

AIRPORT_OUT = ROOT / "data" / "airport"
NON_AIRPORT_OUT = ROOT / "data" / "non_airport"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff"
}

NEGATIVE_CLASSES = [
    "freeway",
    "railway",
    "railway_station",
    "parking_lot",
    "industrial_area",
    "harbor",
    "bridge",
    "intersection",
    "rectangular_farmland",
    "commercial_area",
]


def get_images(folder):
    return [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def find_class_folder(class_name):
    candidates = []

    for folder in SOURCE.rglob("*"):
        if folder.is_dir() and folder.name.lower() == class_name.lower():

            images = get_images(folder)

            if images:
                candidates.append((folder, len(images)))

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find image folder for class: {class_name}"
        )

    # 如果存在多个同名文件夹，选择图片最多的那个
    candidates.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return candidates[0][0]


# -------------------------
# 清空旧数据
# -------------------------

if AIRPORT_OUT.exists():
    shutil.rmtree(AIRPORT_OUT)

if NON_AIRPORT_OUT.exists():
    shutil.rmtree(NON_AIRPORT_OUT)

AIRPORT_OUT.mkdir(parents=True)
NON_AIRPORT_OUT.mkdir(parents=True)


# -------------------------
# Airport positive samples
# -------------------------

airport_folder = find_class_folder("airport")

airport_images = get_images(airport_folder)

print("Airport source:")
print(airport_folder)

print(
    "Airport images found:",
    len(airport_images)
)

for i, image_path in enumerate(airport_images):

    new_name = (
        f"airport_{i:04d}"
        f"{image_path.suffix.lower()}"
    )

    shutil.copy2(
        image_path,
        AIRPORT_OUT / new_name
    )


# -------------------------
# Non-airport negative samples
# -------------------------

random.seed(42)

for class_name in NEGATIVE_CLASSES:

    folder = find_class_folder(class_name)

    images = get_images(folder)

    print(
        f"{class_name}:",
        len(images)
    )

    selected = random.sample(
        images,
        min(70, len(images))
    )

    for i, image_path in enumerate(selected):

        new_name = (
            f"{class_name}_{i:03d}"
            f"{image_path.suffix.lower()}"
        )

        shutil.copy2(
            image_path,
            NON_AIRPORT_OUT / new_name
        )


airport_count = len(
    get_images(AIRPORT_OUT)
)

non_airport_count = len(
    get_images(NON_AIRPORT_OUT)
)

print()
print("======================")
print("Dataset preparation finished")
print("======================")

print("Airport:", airport_count)
print("Non-airport:", non_airport_count)
print("Total:", airport_count + non_airport_count)