# Foundations

## Purpose

This document contains the minimum theory reused throughout the project. Read over some to know the gist of it.

It is intentionally concise. Deeper derivations and model-specific details are kept in the corresponding documentation files.

```text
Foundations
    ↓
Logistic Regression
    ↓
Softmax Regression
    ↓
MLP
    ↓
CNN
    ↓
Vision Transformer
```

---

# 1. Supervised Classification

In supervised learning, a model learns from input-label pairs:

```text
Input X
  ↓
Model
  ↓
Prediction
  ↓
Compare with target y
```

For classification, the target is a discrete class.

This project uses:

```text
Binary classification
→ Logistic Regression

Multiclass classification
→ Softmax Regression and later models
```

For MNIST:

```text
Input  = 28 × 28 grayscale image
Target = digit 0–9
```

After flattening:

$$
X \in \mathbb{R}^{N \times 784}
$$

where $N$ is the number of samples.

---

# 2. Linear Model

A basic linear model computes:

$$
Z = XW + b
$$

where:

- $X$ = input features
- $W$ = learned weights
- $b$ = learned bias
- $Z$ = output scores / logits

The weights determine how input features contribute to the output.

The bias allows the decision function to shift instead of being constrained to pass through the origin.

For MNIST Softmax Regression:

```text
X : N × 784
W : 784 × 10
b : 10
Z : N × 10
```

Each row of $Z$ contains one score for each digit class.

---

# 3. Logits and Probabilities

The output of a linear layer is called a **logit**.

Logits are scores, not probabilities.

```text
Linear model
    ↓
Logits
    ↓
Probability transformation
```

For binary classification, Logistic Regression uses Sigmoid:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

For multiclass classification, Softmax converts the logits into class probabilities:

$$
p_k =
\frac{e^{z_k}}
{\sum_j e^{z_j}}
$$

Prediction is obtained from the largest output score:

```text
prediction = argmax(logits)
```

For multiclass classification, `argmax(logits)` and `argmax(softmax(logits))` give the same class because Softmax preserves ordering.

---

# 4. Loss Functions

A loss function measures how incorrect the model prediction is.

## Binary Cross Entropy

Used by Logistic Regression:

$$
L =
-\left[
y\log(p)
+
(1-y)\log(1-p)
\right]
$$

## Multiclass Cross Entropy

Used by Softmax Regression and later classifiers:

$$
L = -\log(p_y)
$$

where $p_y$ is the predicted probability of the correct class.

Training aims to minimize the average loss over the training data.

---

# 5. Gradients and Optimization

A gradient measures how the loss changes with respect to a parameter.

Examples:

$$
\frac{\partial L}{\partial W}
$$

$$
\frac{\partial L}{\partial b}
$$

Gradient Descent updates the parameters in the direction that reduces the loss:

$$
W \leftarrow W - \eta \frac{\partial L}{\partial W}
$$

$$
b \leftarrow b - \eta \frac{\partial L}{\partial b}
$$

where $\eta$ is the learning rate.

The basic training cycle is:

```text
Forward pass
    ↓
Compute loss
    ↓
Compute gradients
    ↓
Update parameters
    ↓
Repeat
```

This same structure remains even as the models become more complex.

---

# 6. Train, Validation, and Test

The dataset is split according to purpose:

```text
Training set
→ learn model parameters

Validation set
→ compare hyperparameters and model choices

Test set
→ final evaluation
```

The test set should not be used to make training or model-selection decisions.

---

# 7. Batches and Epochs

Training data is usually divided into mini-batches.

Important terms:

- **Batch** — a subset of training samples
- **Batch size** — number of samples in one batch
- **Iteration / step** — one parameter update
- **Epoch** — one complete pass through the training set

Conceptually:

```text
Epoch
 ├── Batch 1 → update
 ├── Batch 2 → update
 ├── Batch 3 → update
 └── ...
```

Mini-batch training is used throughout the MNIST stages.

---

# 8. Logistic Regression → Softmax Regression

Logistic Regression performs binary classification:

```text
Input
 ↓
Linear
 ↓
Sigmoid
 ↓
Probability
 ↓
Binary Cross Entropy
```

Softmax Regression generalizes the same idea to multiple classes:

```text
Input
 ↓
Linear
 ↓
Softmax
 ↓
Class probabilities
 ↓
Cross Entropy
```

For MNIST:

```text
784 input features
       ↓
Linear
       ↓
10 logits
       ↓
Softmax
       ↓
10 probabilities
```

The main change is the output representation:

```text
Binary
Sigmoid
one probability

        ↓

Multiclass
Softmax
one probability per class
```

The overall training process remains the same.

---

# 9. Reusable Learning Pipeline

The project treats models as combinations of reusable components:

```text
Data
 ↓
Preprocessing
 ↓
Representation / Model
 ↓
Output
 ↓
Loss
 ↓
Optimization
 ↓
Evaluation
```

Later models mainly change how the input is represented and transformed.

For example:

```text
Softmax Regression
Linear representation

MLP
Nonlinear hidden representation

CNN
Spatial representation

Vision Transformer
Patch / attention representation
```

This makes it easier to identify what is reused, replaced, or added at each stage.

---

# 10. Core Takeaways

The essential concepts from this document are:

1. Supervised classification learns from input-label pairs.
2. A linear model computes $Z = XW + b$.
3. Logits are scores, not probabilities.
4. Sigmoid is used for binary probability output.
5. Softmax is used for multiclass probability output.
6. Loss measures prediction error.
7. Gradients show how parameters affect the loss.
8. Gradient-based optimization updates parameters to reduce loss.
9. Training, validation, and test sets have different roles.
10. Mini-batch training performs repeated parameter updates.
11. Logistic Regression extends naturally into Softmax Regression.
12. The same training pipeline is reused as the project moves toward MLP, CNN, and Vision Transformer.

The central learning loop is:

```text
Input
  ↓
Model
  ↓
Logits
  ↓
Loss
  ↓
Gradients
  ↓
Parameter update
```

Everything later in the project builds on this structure.
