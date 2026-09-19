from pathlib import Path
import argparse
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the checkpoint created by our training script.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Match the validation preprocessing used during training.
    transform = transforms.Compose([
        transforms.Resize(tuple(checkpoint["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["mean"],
            std=checkpoint["std"],
        ),
    ])

    dataset = datasets.ImageFolder(
        ROOT / "data" / "split" / "test",
        transform=transform,
    )

    if dataset.class_to_idx != checkpoint["class_to_idx"]:
        raise ValueError("Test labels do not match checkpoint labels.")

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    # Restore the trained model without downloading pretrained weights.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_airport_probabilities = []

    airport_index = dataset.class_to_idx["airport"]
    non_airport_index = dataset.class_to_idx["non_airport"]

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            predictions = outputs.argmax(dim=1)
            probabilities = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_airport_probabilities.extend(
                probabilities[:, airport_index].cpu().tolist()
            )

    # Rows represent true labels; columns represent predicted labels.
    confusion = torch.zeros((2, 2), dtype=torch.long)

    for actual, predicted in zip(all_labels, all_predictions):
        confusion[actual, predicted] += 1

    # Treat airport as the positive class.
    tp = confusion[airport_index, airport_index].item()
    fn = confusion[airport_index, non_airport_index].item()
    fp = confusion[non_airport_index, airport_index].item()
    tn = confusion[non_airport_index, non_airport_index].item()

    accuracy = (tp + tn) / len(dataset)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print("Checkpoint:", checkpoint_path)
    print("Selected epoch:", checkpoint["epoch"])
    print("Test images:", len(dataset))
    print(f"Test loss: {total_loss / len(dataset):.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Airport precision: {precision:.2%}")
    print(f"Airport recall: {recall:.2%}")
    print(f"Airport F1: {f1:.2%}")

    print("\nConfusion matrix:")
    print("Rows = actual, columns = predicted")
    print("Class order:", dataset.classes)
    print(confusion.numpy())

    print(f"\nMissed airports: {fn}")
    print(f"False airport detections: {fp}")

    # Save predictions in the same order as the test dataset.
    predictions_path = checkpoint_path.parent / "test_predictions.csv"

    with predictions_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "image",
            "actual",
            "predicted",
            "airport_probability",
            "correct",
        ])

        for index, (actual, predicted, probability) in enumerate(zip(
            all_labels,
            all_predictions,
            all_airport_probabilities,
        )):
            image_path = Path(dataset.samples[index][0])
            writer.writerow([
                str(image_path.relative_to(ROOT)),
                dataset.classes[actual],
                dataset.classes[predicted],
                probability,
                actual == predicted,
            ])

    print("\nPredictions saved:", predictions_path)


if __name__ == "__main__":
    main()