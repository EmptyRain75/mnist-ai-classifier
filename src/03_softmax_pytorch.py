import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_mnist(batch_size=32, val_size=10000, seed=42):
    """Create reproducible MNIST train/validation/test DataLoaders."""
    transform = transforms.ToTensor()

    full_train = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class SoftmaxRegression(nn.Module):
    """Linear MNIST classifier. CrossEntropyLoss handles Softmax internally."""

    def __init__(self, input_size=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


def train_model(model, train_loader, criterion, optimizer, n_epochs=5):
    """Train the model and return elapsed time."""
    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1:02d}/{n_epochs} | Loss: {average_loss:.4f}")

    return time.perf_counter() - start_time


def evaluate(model, data_loader):
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    batch_size = 32
    learning_rate = 0.1
    n_epochs = 5
    seed = 42

    torch.manual_seed(seed)

    train_loader, val_loader, test_loader = load_mnist(
        batch_size=batch_size, seed=seed
    )

    model = SoftmaxRegression()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print("Training PyTorch Softmax Regression...")
    print(f"Learning rate    : {learning_rate}")
    print(f"Epochs           : {n_epochs}")
    print(f"Batch size       : {batch_size}")
    print(f"Trainable params : {count_parameters(model)}\n")

    training_time = train_model(
        model, train_loader, criterion, optimizer, n_epochs=n_epochs
    )

    train_accuracy = evaluate(model, train_loader)
    val_accuracy = evaluate(model, val_loader)
    test_accuracy = evaluate(model, test_loader)

    print("\nResults")
    print("-------")
    print(f"Training time      : {training_time:.2f} s")
    print(f"Training accuracy  : {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy      : {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
