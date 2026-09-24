import time
import numpy as np


def make_binary_dataset(n_samples=1000, seed=42):
    """Create a simple reproducible 2D binary-classification dataset."""
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0

    class_0 = rng.normal(loc=(-2.0, -2.0), scale=1.2, size=(n0, 2))
    class_1 = rng.normal(loc=(2.0, 2.0), scale=1.2, size=(n1, 2))

    X = np.vstack((class_0, class_1)).astype(np.float64)
    y = np.concatenate((np.zeros(n0), np.ones(n1))).astype(np.float64)

    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def train_test_split(X, y, test_size=0.2, seed=42):
    """Split arrays into train and test sets."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(X_train, X_test):
    """Standardize features using training-set statistics only."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def sigmoid(z):
    """Compute sigmoid activation."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X, W, b):
    """Return P(y=1|x)."""
    return sigmoid(X @ W + b)


def predict(X, W, b, threshold=0.5):
    """Convert probabilities to binary predictions."""
    return (predict_proba(X, W, b) >= threshold).astype(int)


def binary_cross_entropy(y_true, y_pred):
    """Compute mean Binary Cross Entropy."""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )


def compute_gradients(X, y, probabilities):
    """Compute gradients for W and b."""
    n_samples = X.shape[0]
    error = probabilities - y
    dW = (X.T @ error) / n_samples
    db = np.mean(error)
    return dW, db


def update_parameters(W, b, dW, db, learning_rate):
    """Perform one Gradient Descent update."""
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b


def train_logistic_regression(X, y, learning_rate=0.1, num_iterations=1000):
    """Train Logistic Regression with full-batch Gradient Descent."""
    W = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    loss_history = []

    for _ in range(num_iterations):
        probabilities = predict_proba(X, W, b)
        loss_history.append(binary_cross_entropy(y, probabilities))
        dW, db = compute_gradients(X, y, probabilities)
        W, b = update_parameters(W, b, dW, db, learning_rate)

    return W, b, loss_history


def classification_metrics(y_true, y_pred):
    """Return accuracy, precision, recall, and F1."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = np.mean(y_true == y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    learning_rate = 0.1
    num_iterations = 1000
    seed = 42

    X, y = make_binary_dataset(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, seed=seed
    )
    X_train, X_test = standardize_train_test(X_train, X_test)

    print("Training Logistic Regression...")
    print(f"Learning rate : {learning_rate}")
    print(f"Iterations    : {num_iterations}\n")

    start_time = time.perf_counter()
    W, b, loss_history = train_logistic_regression(
        X_train,
        y_train,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
    )
    training_time = time.perf_counter() - start_time

    train_metrics = classification_metrics(y_train, predict(X_train, W, b))
    test_metrics = classification_metrics(y_test, predict(X_test, W, b))

    print("Results")
    print("-------")
    print(f"Training time : {training_time:.3f} s")
    print(f"Final loss    : {loss_history[-1]:.4f}")
    print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Test accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall   : {test_metrics['recall']:.4f}")
    print(f"Test F1       : {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
