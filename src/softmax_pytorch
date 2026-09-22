import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(batch_size=32):
    """
    Load MNIST and create training and test DataLoaders.
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def create_model():
    """
    Create a linear Softmax Regression model for MNIST.
    """
    return nn.Linear(784, 10)


def train_model(
    model,
    train_loader,
    learning_rate=0.1,
    n_epochs=5
):
    """
    Train the model using Cross Entropy and SGD.
    """
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    start_time = time.perf_counter()

    for epoch in range(n_epochs):

        for images, labels in train_loader:

            # Flatten MNIST images:
            # (batch, 1, 28, 28) -> (batch, 784)
            images = images.view(images.size(0), -1)

            # Forward pass
            logits = model(images)

            # Loss
            loss = criterion(logits, labels)

            # Clear previous gradients
            optimizer.zero_grad()

            # Compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"Loss: {loss.item():.4f}"
        )

    training_time = time.perf_counter() - start_time

    return training_time


def evaluate(model, data_loader):
    """
    Compute classification accuracy.
    """
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.view(images.size(0), -1)

            logits = model(images)

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


def main():
    batch_size = 32
    learning_rate = 0.1
    n_epochs = 5

    torch.manual_seed(42)

    train_loader, test_loader = load_mnist(
        batch_size=batch_size
    )

    model = create_model()

    print("Training PyTorch Softmax Regression...")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}\n")

    training_time = train_model(
        model,
        train_loader,
        learning_rate=learning_rate,
        n_epochs=n_epochs
    )

    test_accuracy = evaluate(
        model,
        test_loader
    )

    print("\nResults")
    print("-------")
    print(
        f"Training time: {training_time:.2f} s"
    )
    print(
        f"Test accuracy: {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
