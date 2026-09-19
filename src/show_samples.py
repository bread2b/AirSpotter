from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

ROOT = Path(__file__).resolve().parent.parent

airport_folder = ROOT / "data" / "airport"
non_airport_folder = ROOT / "data" / "non_airport"

airport_images = list(airport_folder.glob("*"))
non_airport_images = list(non_airport_folder.glob("*"))

random.seed(42)

airport_samples = random.sample(
    airport_images,
    8
)

non_airport_samples = random.sample(
    non_airport_images,
    8
)

samples = airport_samples + non_airport_samples

fig, axes = plt.subplots(
    4,
    4,
    figsize=(12, 12)
)

for i, (ax, path) in enumerate(
    zip(axes.flat, samples)
):

    image = Image.open(path)

    ax.imshow(image)

    if i < 8:
        label = "Airport"
    else:
        label = "Non-Airport"

    ax.set_title(
        f"{label}\n{path.name}",
        fontsize=8
    )

    ax.axis("off")

plt.tight_layout()
plt.show()