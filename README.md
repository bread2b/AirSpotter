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
│   └── predict.py          # Predict one image / 单图预测
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
python src/predict.py --checkpoint $checkpoint --image "path/to/image.jpg"
```

Replace `path/to/image.jpg` with your image path. Evaluation needs the local test dataset; single-image prediction only needs the model and your image.

中文备注：把图片路径换成自己的。评估需要本地测试集，单图预测不需要下载整套数据集。模型给出的分数不保证判断正确。

## Saved Results / 已保存的结果

The run in `outputs/20260919_120214_330283/` contains:

- `best_model.pth`: Best model, selected at epoch 3 / 第 3 轮选出的最佳模型。
- `history.csv`: Training and validation results for 10 epochs / 10 轮训练记录。
- `test_predictions.csv`: Predictions for all 210 test images / 每张测试图片的预测结果。

Next: test more images from different sources and inspect mistakes.

下一步：用更多不同来源的图片测试，看看模型在哪些情况下容易判断错误。

## Test Commands / 测试指令

Run from the project root in PowerShell. No environment activation is needed.

在项目根目录打开 PowerShell，直接运行，无需先激活环境。

**Evaluate the test set / 测试整套测试集**

Requires the prepared `data/split/test/` folder. Prints accuracy and saves predictions to `test_predictions.csv` beside the model, replacing the previous file.

需要本地已有测试集。运行后显示准确率，并覆盖模型所在文件夹中的预测结果表。

```powershell
.\.venv\Scripts\python.exe src/evaluate.py --checkpoint "outputs/20260919_120214_330283/best_model.pth"
```

**Test your own image / 测试自己的图片**

Replace `C:\path\to\image.jpg` with your image's actual path. Prints the predicted class and scores.

把下面的图片路径换成自己的实际路径，运行后会显示是否为机场及模型分数。

```powershell
.\.venv\Scripts\python.exe src/predict.py --checkpoint "outputs/20260919_120214_330283/best_model.pth" --image "C:\path\to\image.jpg"
```
