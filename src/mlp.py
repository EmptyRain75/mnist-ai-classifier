import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MLP(nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden_size=128,
        num_classes=10
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


def load_mnist(
    batch_size=32,
    val_size=10000,
    seed=42
):
    transform = transforms.ToTensor()

    full_train_dataset = datasets.MNIST(
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

    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    n_epochs=10
):
    loss_history = []

    start_time = time.perf_counter()

    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for images, labels in train_loader:
            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        average_loss = (
            epoch_loss / len(train_loader)
        )

        loss_history.append(
            average_loss
        )

        print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"Loss: {average_loss:.4f}"
        )

    training_time = (
        time.perf_counter() - start_time
    )

    return loss_history, training_time


def evaluate(model, data_loader):
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in data_loader:
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


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main():
    batch_size = 32
    learning_rate = 0.1
    n_epochs = 10
    hidden_size = 128

    torch.manual_seed(42)

    train_loader, val_loader, test_loader = load_mnist(
        batch_size=batch_size
    )

    model = MLP(
        hidden_size=hidden_size
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate
    )

    print("Training MLP...")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(
        f"Trainable parameters: "
        f"{count_parameters(model)}\n"
    )

    _, training_time = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        n_epochs=n_epochs
    )

    train_accuracy = evaluate(
        model,
        train_loader
    )

    val_accuracy = evaluate(
        model,
        val_loader
    )

    test_accuracy = evaluate(
        model,
        test_loader
    )

    print("\nResults")
    print("-------")

    print(
        f"Training time      : "
        f"{training_time:.2f} s"
    )

    print(
        f"Training accuracy  : "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Test accuracy      : "
        f"{test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
