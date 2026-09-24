import time
from pathlib import Path

import numpy as np
from torchvision import datasets


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_mnist(val_size=10000, seed=42):
    """Load MNIST and return flattened NumPy train/val/test arrays."""
    train_dataset = datasets.MNIST(root=DATA_DIR, train=True, download=True)
    test_dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True)

    X_full = train_dataset.data.numpy()
    y_full = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy()
    y_test = test_dataset.targets.numpy()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_full))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train = X_full[train_idx]
    y_train = y_full[train_idx]
    X_val = X_full[val_idx]
    y_val = y_full[val_idx]

    X_train = X_train.reshape(len(X_train), -1).astype(np.float32) / 255.0
    X_val = X_val.reshape(len(X_val), -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32) / 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test


def softmax(logits):
    """Compute numerically stable Softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def predict_proba(X, W, b):
    """Return class probabilities."""
    return softmax(X @ W + b)


def predict(X, W, b):
    """Return predicted classes."""
    return np.argmax(X @ W + b, axis=1)


def cross_entropy(y, probabilities):
    """Compute mean multiclass Cross Entropy."""
    correct_probs = probabilities[np.arange(len(y)), y]
    correct_probs = np.clip(correct_probs, 1e-12, 1.0)
    return -np.mean(np.log(correct_probs))


def one_hot_encode(y, n_classes=10):
    """One-hot encode class labels."""
    return np.eye(n_classes, dtype=np.float32)[y]


def compute_gradients(X, y, probabilities, n_classes=10):
    """Compute gradients for W and b."""
    n_samples = X.shape[0]
    targets = one_hot_encode(y, n_classes)
    d_logits = probabilities - targets
    dW = (X.T @ d_logits) / n_samples
    db = np.mean(d_logits, axis=0)
    return dW, db


def update_parameters(W, b, dW, db, learning_rate):
    """Perform one Gradient Descent update."""
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b


def train_softmax_minibatch(
    X,
    y,
    n_classes=10,
    learning_rate=0.1,
    n_epochs=10,
    batch_size=32,
    seed=42,
):
    """Train Softmax Regression with mini-batch Gradient Descent."""
    n_samples, n_features = X.shape
    W = np.zeros((n_features, n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        indices = rng.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]

            probabilities = predict_proba(X_batch, W, b)
            dW, db = compute_gradients(X_batch, y_batch, probabilities, n_classes)
            W, b = update_parameters(W, b, dW, db, learning_rate)

        train_loss = cross_entropy(y, predict_proba(X, W, b))
        print(f"Epoch {epoch + 1:02d}/{n_epochs} | Loss: {train_loss:.4f}")

    training_time = time.perf_counter() - start_time
    return W, b, training_time


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)


def main():
    learning_rate = 0.1
    n_epochs = 10
    batch_size = 32
    n_classes = 10
    seed = 42

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(seed=seed)

    print("Training NumPy Softmax Regression...")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs       : {n_epochs}")
    print(f"Batch size   : {batch_size}\n")

    W, b, training_time = train_softmax_minibatch(
        X_train,
        y_train,
        n_classes=n_classes,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
    )

    train_accuracy = accuracy(y_train, predict(X_train, W, b))
    val_accuracy = accuracy(y_val, predict(X_val, W, b))
    test_accuracy = accuracy(y_test, predict(X_test, W, b))
    final_loss = cross_entropy(y_train, predict_proba(X_train, W, b))

    print("\nResults")
    print("-------")
    print(f"Training time      : {training_time:.2f} s")
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Training accuracy  : {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy      : {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
