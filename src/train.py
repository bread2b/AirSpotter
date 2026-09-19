from pathlib import Path
from datetime import datetime
import csv
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "split"

SEED = 42
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    # Disable gradient tracking during validation.
    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_images += batch_size

    return (
        total_loss / total_images,
        total_correct / total_images,
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check your environment.")

    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")

    # Create a separate output folder for each training run.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = ROOT / "outputs" / run_name

    weights = models.ResNet18_Weights.DEFAULT

    # Preserve the full image and apply random flips during training.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])

    # Use deterministic preprocessing during validation.
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])

    train_dataset = datasets.ImageFolder(
        DATA / "train",
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        DATA / "val",
        transform=val_transform,
    )

    expected_mapping = {"airport": 0, "non_airport": 1}

    if (
        train_dataset.class_to_idx != expected_mapping
        or val_dataset.class_to_idx != expected_mapping
    ):
        raise ValueError("Unexpected class folders or label mapping.")

    # Start with a single-process loader for simplicity on Windows.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Classes:", train_dataset.class_to_idx)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Loading pretrained ResNet18...", flush=True)

    # Replace the original classifier with a two-class classifier.
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Fine-tune all model parameters with a small learning rate.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output_dir / "best_model.pth"
    history_path = output_dir / "history.csv"

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_accuracy = 0.0

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
        ])

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_accuracy = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_loss, val_accuracy = run_epoch(
                model, val_loader, criterion, device
            )

            writer.writerow([
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            ])
            file.flush()

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_accuracy:.2%} | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_accuracy:.2%}",
                flush=True,
            )

            # Select the best checkpoint using validation loss only.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_val_accuracy = val_accuracy

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "architecture": "resnet18",
                    "class_to_idx": train_dataset.class_to_idx,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "seed": SEED,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "image_size": [224, 224],
                    "mean": weights.transforms().mean,
                    "std": weights.transforms().std,
                }, checkpoint_path)

                print("  Saved best model.", flush=True)

    print("\nTraining finished.")
    print("Best epoch:", best_epoch)
    print(f"Validation accuracy at best epoch: {best_val_accuracy:.2%}")
    print("Model:", checkpoint_path)
    print("History:", history_path)


if __name__ == "__main__":
    main()