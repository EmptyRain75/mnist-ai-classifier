import numpy as np


def sigmoid(z):
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-z))


def predict_proba(X, W, b):
    """Compute predicted probabilities for binary classification."""
    z = X @ W + b
    return sigmoid(z)


def predict(X, W, b, threshold=0.5):
    """Convert predicted probabilities into binary class predictions."""
    probabilities = predict_proba(X, W, b)
    return (probabilities >= threshold).astype(int)


def binary_cross_entropy(y_true, y_pred):
    """Compute mean Binary Cross Entropy loss."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return loss


def compute_gradients(X, y, p):
    """Compute analytical gradients with respect to W and b."""
    N = X.shape[0]
    error = p - y

    dW = (X.T @ error) / N
    db = np.mean(error)

    return dW, db


def update_parameters(W, b, dW, db, learning_rate):
    """Perform one gradient descent update."""
    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b


def train_logistic_regression(
    X,
    y,
    learning_rate=0.1,
    num_iterations=1000,
):
    """Train Logistic Regression using full-batch gradient descent."""
    W = np.zeros(X.shape[1])
    b = 0.0

    loss_history = []

    for _ in range(num_iterations):
        # Forward pass
        p = predict_proba(X, W, b)

        # Loss
        loss = binary_cross_entropy(y, p)

        # Gradients
        dW, db = compute_gradients(X, y, p)

        # Gradient descent update
        W, b = update_parameters(
            W,
            b,
            dW,
            db,
            learning_rate,
        )

        loss_history.append(loss)

    return W, b, loss_history


def accuracy(y_true, y_pred):
    """Compute classification accuracy."""
    return np.mean(y_true == y_pred)


def classification_metrics(y_true, y_pred):
    """Compute binary classification metrics."""
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
    }
