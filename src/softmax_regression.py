import time
import numpy as np
from tensorflow.keras.datasets import mnist


def load_mnist(val_size=10000, seed=42):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_train))

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_val = X_val.reshape(X_val.shape[0], -1).astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test


def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def one_hot_encode(y, n_classes=10):
    return np.eye(n_classes)[y]


def cross_entropy(y, P):
    n_samples = len(y)
    correct_class_probs = P[np.arange(n_samples), y]
    correct_class_probs = np.clip(correct_class_probs, 1e-12, 1.0)
    return -np.mean(np.log(correct_class_probs))


def compute_gradients(X, y, P, n_classes=10):
    n_samples = X.shape[0]
    Y = one_hot_encode(y, n_classes)
    dZ = P - Y
    dW = (X.T @ dZ) / n_samples
    db = np.mean(dZ, axis=0)
    return dW, db


def update_parameters(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b


def predict_proba(X, W, b):
    return softmax(X @ W + b)


def predict(X, W, b):
    return np.argmax(predict_proba(X, W, b), axis=1)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred, n_classes=10):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, predicted_label in zip(y_true, y_pred):
        matrix[true_label, predicted_label] += 1
    return matrix


def train_softmax_regression(
    X,
    y,
    n_classes=10,
    learning_rate=0.1,
    n_iterations=1000
):
    n_features = X.shape[1]
    W = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)
    loss_history = []

    for _ in range(n_iterations):
        Z = X @ W + b
        P = softmax(Z)
        loss_history.append(cross_entropy(y, P))

        dW, db = compute_gradients(X, y, P, n_classes)
        W, b = update_parameters(W, b, dW, db, learning_rate)

    return W, b, loss_history


def train_softmax_minibatch(
    X,
    y,
    n_classes=10,
    learning_rate=0.1,
    n_epochs=50,
    batch_size=32,
    seed=42
):
    n_samples, n_features = X.shape

    W = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)

    rng = np.random.default_rng(seed)
    start_time = time.perf_counter()

    for _ in range(n_epochs):
        indices = rng.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            P = softmax(X_batch @ W + b)

            dW, db = compute_gradients(
                X_batch,
                y_batch,
                P,
                n_classes
            )

            W, b = update_parameters(
                W,
                b,
                dW,
                db,
                learning_rate
            )

    training_time = time.perf_counter() - start_time
    return W, b, training_time


def main():
    # Final mini-batch configuration from the notebook experiments.
    learning_rate = 0.1
    n_epochs = 50
    batch_size = 32
    n_classes = 10

    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

    print("Training Softmax Regression...")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}\n")

    W, b, training_time = train_softmax_minibatch(
        X_train,
        y_train,
        n_classes=n_classes,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size
    )

    train_pred = predict(X_train, W, b)
    val_pred = predict(X_val, W, b)
    test_pred = predict(X_test, W, b)

    train_acc = accuracy(y_train, train_pred)
    val_acc = accuracy(y_val, val_pred)
    test_acc = accuracy(y_test, test_pred)

    train_probs = predict_proba(X_train, W, b)
    final_loss = cross_entropy(y_train, train_probs)

    print("Results")
    print("-------")
    print(f"Training time      : {training_time:.2f} s")
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Training accuracy  : {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy      : {test_acc:.4f}")

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_pred, n_classes))


if __name__ == "__main__":
    main()
