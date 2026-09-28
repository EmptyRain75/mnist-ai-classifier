import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ----------------------------
# Data
# ----------------------------

def load_mnist(batch_size=32, val_size=10000, seed=42):
    """Create reproducible MNIST train/validation/test DataLoaders."""
    transform = transforms.ToTensor()

    full_train = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = len(full_train) - val_size

    train_dataset, val_dataset = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


# ----------------------------
# Model
# ----------------------------

class CNN(nn.Module):
    """Basic CNN for MNIST."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc = nn.Linear(
            64 * 7 * 7,
            num_classes,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


# ----------------------------
# Training + Evaluation
# ----------------------------

def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    n_epochs=10,
):
    """Train the CNN and return elapsed time."""
    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1:02d}/{n_epochs} | "
            f"Loss: {average_loss:.4f}"
        )

    return time.perf_counter() - start_time


def evaluate(model, data_loader, device):
    """Compute classification accuracy."""
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def count_parameters(model):
    """Count trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ----------------------------
# One-click Run
# ----------------------------

def main():
    batch_size = 32
    learning_rate = 0.1
    n_epochs = 10
    seed = 42

    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader, test_loader = load_mnist(
        batch_size=batch_size,
        seed=seed,
    )

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    print("Training CNN...")
    print(f"Device            : {device}")
    print(f"Learning rate     : {learning_rate}")
    print(f"Epochs            : {n_epochs}")
    print(f"Batch size        : {batch_size}")
    print(f"Trainable params  : {count_parameters(model)}\n")

    training_time = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        n_epochs=n_epochs,
    )

    train_accuracy = evaluate(
        model,
        train_loader,
        device,
    )

    val_accuracy = evaluate(
        model,
        val_loader,
        device,
    )

    test_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    print("\nResults")
    print("-------")
    print(f"Training time      : {training_time:.2f} s")
    print(f"Training accuracy  : {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy      : {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
