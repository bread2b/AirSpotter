# AirSpotter V1

A Python project that classifies satellite images as **airport** or **non-airport**, using ResNet18.

中文备注：判断卫星图片中有没有机场，目前支持训练、测试和单张图片预测，不支持画框定位。

## Progress / 当前进度

The first model has been trained on a dataset of 700 airport and 700 non-airport images. The split contains 980 training, 210 validation, and 210 test images.

**Test accuracy: 97.62% (205/210 correct).** This result applies to the current RESISC45-based test set; performance on images from other sources still needs testing.

中文备注：第一版已完成，测试集答对 205 张、答错 5 张。这个成绩不代表所有卫星图片上的准确率。

## Files / 文件说明

```text
AirSpotter/
├── src/
│   ├── prepare_dataset.py  # Prepare images / 整理图片
│   ├── show_samples.py     # Preview samples / 预览图片
│   ├── split_dataset.py    # Split the dataset / 划分数据
│   ├── train.py            # Train ResNet18 / 训练模型
│   ├── evaluate.py         # Test the model / 评估模型
│   ├── predict.py          # Predict one image / 单图预测
│   └── scan_large_image.py # Scan a large map / 扫描大地图
├── outputs/                # Saved model and training results / 模型和训练结果
├── results/                # Additional charts / 其他图表
├── notebooks/              # Jupyter notebooks / 笔记本
├── data/                   # Prepared images and splits / 整理后的图片
├── RESISC45/               # Original dataset / 原始数据集
├── .venv/                  # Local Python environment / 本地环境
├── .gitignore
└── README.md
```

Code, model weights, and training results can be pushed to GitHub. `RESISC45/`, images in `data/`, and `.venv/` stay local.

中文备注：上传代码、模型和结果，不上传原始数据集、训练图片和虚拟环境。

## Run / 使用方法

Use Python 3.12 with PyTorch, torchvision, NumPy, Matplotlib, and Pillow. Jupyter is optional. Training currently requires an NVIDIA GPU and a CUDA-enabled PyTorch installation; evaluation and prediction also support CPU.

中文备注：本地环境已配置好。换电脑需要重新安装环境；当前训练脚本需要 NVIDIA 显卡，预测可以用 CPU。

From the project root in PowerShell / 在项目根目录运行：

```powershell
.\.venv\Scripts\Activate.ps1
```

To prepare data and train a new model / 整理数据并训练新模型：

```powershell
python src/prepare_dataset.py
python src/show_samples.py
python src/split_dataset.py
python src/train.py
```

`prepare_dataset.py` replaces the two prepared image folders. `split_dataset.py` expects 700 images per class and will stop if `data/split/` already exists. Training may download pretrained weights on its first run.

中文备注：整理脚本会清空并重建两类图片文件夹；已有数据划分时不用重复运行。使用现成模型预测时，可跳过以上四步。

To use the saved model / 使用已保存的模型：

```powershell
$checkpoint = "outputs/20260919_120214_330283/best_model.pth"
python src/evaluate.py --checkpoint $checkpoint
python src/predict.py --checkpoint $checkpoint --image "data/manual/demo.jpg"
```

Place your own images in `data/manual/` and replace `demo.jpg` with the filename you want to test. Evaluation needs the local test dataset; single-image prediction only needs the model and your image.

中文备注：手动测试的图片统一放在 `data/manual/`，把 `demo.jpg` 换成实际文件名。评估需要本地测试集，单图预测只需要模型和待测图片。模型给出的分数不保证判断正确。

## Saved Results / 已保存的结果

The run in `outputs/20260919_120214_330283/` contains:

- `best_model.pth`: Best model, selected at epoch 3 / 第 3 轮选出的最佳模型。
- `history.csv`: Training and validation results for 10 epochs / 10 轮训练记录。
- `test_predictions.csv`: Predictions for all 210 test images / 每张测试图片的预测结果。

Next: test more images from different sources and inspect mistakes.

下一步：用更多不同来源的图片测试，看看模型在哪些情况下容易判断错误。

## Large Map Scan / 大地图扫描

The scanner uses a 48 by 24 base grid and scans individual cells, 3 by 3 groups, and 4 by 4 groups. It repeats the scan at four half-cell offsets, combines the three scale scores, merges adjacent cells, and confirms each candidate using an expanded crop. This is an experimental use of the classifier and is not yet reliable across different maps.

扫描器先把地图划分成 48 列、24 行，分别扫描单格、3×3 和 4×4 区域，再用四种半格偏移重复扫描。程序综合三种尺度的分数、合并相邻网格，并扩大候选区域进行二次确认。这只是复用分类模型的实验方案，目前在不同地图上的表现还不稳定。

Put the large image in `data/manual/`, then run:

把大地图放入 `data/manual/`，然后运行：

```powershell
.\.venv\Scripts\python.exe src/scan_large_image.py --image "data/manual/large_map.jpg" --checkpoint "outputs/20260919_120214_330283/best_model.pth"
```

The annotated image, final candidate CSV, and complete grid-score CSV are saved in `results/`.

带方框的图片、最终候选坐标和全部网格分数会保存在 `results/`。

## Test Commands / 测试指令

Run from the project root in PowerShell. No environment activation is needed.

在项目根目录打开 PowerShell，直接运行，无需先激活环境。

Put all images for manual testing in `data/manual/`. Replace `demo.jpg` with the filename you want to test.

所有手动测试图片都放在 `data/manual/`。把 `demo.jpg` 换成要测试的图片文件名。

```powershell
.\.venv\Scripts\python.exe src/predict.py --checkpoint "outputs/20260919_120214_330283/best_model.pth" --image "data/manual/demo.jpg"
```
