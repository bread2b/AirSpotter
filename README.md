# AirSpotter V1

AirSpotter is a beginner-friendly Python project that aims to tell whether a satellite image contains an airport.

## Current Progress

The project folders and Python environment are ready. Two scripts are available to prepare images from the local RESISC45 dataset and preview examples. No model has been built.

## Folder Structure

```text
AirSpotter/
├── .venv/              # Local Python environment
├── RESISC45/           # Original dataset
├── data/
│   ├── airport/        # Images with airports
│   └── non_airport/    # Images without airports
├── src/                # Python code
│   ├── prepare_dataset.py
│   └── show_samples.py
├── notebooks/          # Jupyter notebooks
├── results/            # Charts and results
├── .gitignore
└── README.md
```

The dataset, training images, and `.venv/` stay on your computer and are ignored by Git. Small `.gitkeep` files keep empty project folders visible on GitHub.

## Python Environment

The local environment uses Python 3.12 with NumPy, Matplotlib, Pillow, and Jupyter installed.

From the project folder in PowerShell, activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

To open Jupyter without activating the environment:

```powershell
.\.venv\Scripts\python.exe -m notebook
```

On another computer, install Python 3.12 and create a new environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy matplotlib pillow jupyter
```

## Scripts

- `prepare_dataset.py`: Copies airport images and selects up to 70 images from each of 10 other scene categories into the two `data/` folders. Running it replaces the existing contents of those folders.
- `show_samples.py`: Shows 8 airport and 8 non-airport images in a grid so you can check them visually. Run it after preparing the dataset, with at least 8 images in each folder.

Run these commands from the project folder:

```powershell
.\.venv\Scripts\python.exe src/prepare_dataset.py
.\.venv\Scripts\python.exe src/show_samples.py
```

## Next Steps

1. Check the dataset's usage terms.
2. Prepare images using `prepare_dataset.py`.
3. Preview some images and check their labels.
4. Build a simple model later.
