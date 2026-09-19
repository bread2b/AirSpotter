# AirSpotter V1

AirSpotter is a beginner-friendly Python project that aims to tell whether a satellite image contains an airport.

中文备注：一个适合入门的 Python 项目，用来判断卫星图片中是否有机场。

## Current Progress / 当前进度

The project folders and Python environment are ready. Two scripts are available to prepare images from the local RESISC45 dataset and preview examples. No model has been built.

中文备注：文件夹和 Python 环境已准备好，目前有整理数据和预览图片的两个脚本，还没有开始做模型。

## Folder Structure / 文件夹结构

```text
AirSpotter/
├── .venv/              # Local Python environment / 本地 Python 环境
├── RESISC45/           # Original dataset / 原始数据集
├── data/
│   ├── airport/        # Images with airports / 有机场的图片
│   └── non_airport/    # Images without airports / 没有机场的图片
├── src/                # Python code / Python 代码
│   ├── prepare_dataset.py
│   └── show_samples.py
├── notebooks/          # Jupyter notebooks / 交互式笔记本
├── results/            # Charts and results / 图表和结果
├── .gitignore
└── README.md
```

The dataset, training images, and `.venv/` stay on your computer and are ignored by Git. Small `.gitkeep` files keep empty project folders visible on GitHub.

中文备注：数据集、训练图片和虚拟环境只留在本地，不上传 GitHub。`.gitkeep` 用于保留空文件夹。

## Python Environment / Python 环境

The local environment uses Python 3.12 with NumPy, Matplotlib, Pillow, and Jupyter installed.

中文备注：本地使用 Python 3.12，已安装基础的数据处理、图片查看和笔记本工具。

From the project folder in PowerShell, activate it:

在项目根目录打开 PowerShell，激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

To open Jupyter without activating the environment:

不激活环境也可以直接启动 Jupyter：

```powershell
.\.venv\Scripts\python.exe -m notebook
```

On another computer, install Python 3.12 and create a new environment:

换一台电脑时，先安装 Python 3.12，再新建环境并安装基础包：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy matplotlib pillow jupyter
```

## Scripts / 脚本用途

- `prepare_dataset.py`: Copies airport images and selects up to 70 images from each of 10 other scene categories into the two `data/` folders. Running it replaces the existing contents of those folders.
  中文备注：复制机场图片，并从另外 10 类场景中各选最多 70 张，整理到两个数据文件夹。每次运行都会清空并重新生成这两个文件夹里的内容。
- `show_samples.py`: Shows 8 airport and 8 non-airport images in a grid so you can check them visually. Run it after preparing the dataset, with at least 8 images in each folder.
  中文备注：各展示 8 张机场和非机场图片，方便检查分类是否正确。先整理数据，并确保每类至少有 8 张图片。

Run these commands from the project folder:

在项目根目录依次运行：

```powershell
.\.venv\Scripts\python.exe src/prepare_dataset.py
.\.venv\Scripts\python.exe src/show_samples.py
```

## Next Steps / 下一步

1. Check the dataset's usage terms. / 确认数据集的使用条件。
2. Prepare images using `prepare_dataset.py`. / 运行脚本整理图片。
3. Preview some images and check their labels. / 预览图片，检查分类。
4. Build a simple model later. / 之后再做一个简单模型。
