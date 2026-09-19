# Foundations

## Purpose

This document contains the minimum theoretical foundation needed before implementing Logistic Regression and Softmax Regression from scratch.

The goal is not to relearn all of machine learning, but to establish a clear understanding of the components that will later be reused across the project.

The project follows this progression:

```text
Linear Model
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

# 1. Supervised Learning

Supervised learning is a machine learning setting where the model learns from examples containing:

* **Input** `X`
* **Target/label** `Y`

The model learns a function:

```text
X → Model → Prediction
```

For classification:

```text
Image/features → Model → Class
```

For example:

```text
Input:
[5.1, 3.5, 1.4, 0.2]

Label:
"Setosa"
```

During training, the model compares its prediction with the correct label and adjusts its parameters.

---

# 2. Classification

Classification means predicting a discrete category.

Examples:

```text
Email → Spam / Not Spam
Image → Cat / Dog
MNIST → 0 / 1 / 2 / ... / 9
```

There are two important cases in this project.

### Binary classification

There are two possible classes:

```text
Class 0
Class 1
```

Example:

```text
Pass / Fail
Cat / Dog
Spam / Not Spam
```

This is what we will use for the Logistic Regression experiment.

### Multiclass classification

There are more than two classes:

```text
0, 1, 2, ..., 9
```

This is what we will use for MNIST.

The transition is important:

```text
Binary classification
       ↓
Logistic Regression
       ↓
Multiclass classification
       ↓
Softmax Regression
```

---

# 3. Features and Labels

Suppose we have a dataset with `N` samples and `D` features.

We can represent the input as a matrix:

```text
X ∈ R^(N × D)
```

For example:

```text
X =
[
  x₁₁ x₁₂ x₁₃
  x₂₁ x₂₂ x₂₃
  x₃₁ x₃₂ x₃₃
]
```

There are:

* `N = 3` samples
* `D = 3` features

Each row represents one sample.

The corresponding labels are:

```text
y = [0, 1, 1]
```

For MNIST:

```text
N = number of images
D = 28 × 28 = 784
```

Therefore, after flattening:

```text
X ∈ R^(N × 784)
```

---

# 4. Parameters

A model contains parameters that are learned during training.

For a simple linear model:

```text
z = Wx + b
```

where:

* `W` = weights
* `b` = bias
* `x` = input
* `z` = output/logit

The training process finds values of `W` and `b` that produce useful predictions.

---

# 5. Why Do We Need Bias?

Consider:

```text
z = wx
```

The corresponding decision boundary is forced to pass through the origin.

Adding a bias gives:

```text
z = wx + b
```

Now the model can shift the decision boundary.

For example:

```text
Without bias:

        /
       /
      /
-----/---------
    /
```

With bias:

```text
       /
      /
-----/---------
    /
```

The important idea is:

> **Weights control the orientation/slope, while bias allows the model to shift the decision boundary.**

This becomes especially important when we move from one-dimensional examples to higher-dimensional classification.

---

# 6. Linear Transformation

The basic operation used by our models is:

```text
z = Wx + b
```

For one sample:

```text
x ∈ R^D
W ∈ R^D
b ∈ R
```

For multiple samples, we use matrix multiplication:

```text
Z = XW + b
```

For example, in MNIST:

```text
X : N × 784
W : 784 × 10
b : 10
```

Therefore:

```text
XW : N × 10
```

Each row of `XW` contains the scores for the 10 MNIST classes.

---

# 7. Logits

The values produced by the linear layer are called **logits**.

For example:

```text
z = [2.1, -0.5, 4.2, 0.7, ...]
```

These are not probabilities.

They are simply scores indicating how strongly the model currently associates the input with each class.

For MNIST:

```text
logits:
[2.1, -0.5, 4.2, 0.7, ...]
```

The largest value corresponds to the model's predicted class:

```text
prediction = argmax(logits)
```

However, if we want probabilities, we need a transformation.

For binary classification:

```text
logit → Sigmoid → probability
```

For multiclass classification:

```text
logits → Softmax → probabilities
```

---

# 8. Activation Functions

An activation function transforms the output of a model.

For Logistic Regression, we use the sigmoid function:

```text
σ(z) = 1 / (1 + e^(-z))
```

Its output is between `0` and `1`.

Therefore:

```text
z = 2.0
↓
sigmoid
↓
0.881
```

This can be interpreted as a probability for the positive class.

For example:

```text
P(y = 1 | x) = 0.881
```

---

# 9. Logistic Regression

Logistic Regression combines a linear model with sigmoid:

```text
x
↓
Linear
z = Wx + b
↓
Sigmoid
p = σ(z)
↓
Probability
```

Mathematically:

```text
z = Wx + b

p = σ(z)
```

The prediction can then be obtained using a threshold:

```text
if p ≥ 0.5:
    class = 1
else:
    class = 0
```

The important point is:

> Logistic Regression is still a linear classifier. The sigmoid converts the linear score into a probability.

---

# 10. Loss Function

During training, we need a way to measure how wrong the model is.

This is the **loss function**.

For binary Logistic Regression, we use Binary Cross Entropy:

```text
L = -[y log(p) + (1-y) log(1-p)]
```

where:

* `y` = true label
* `p` = predicted probability

Example:

```text
True label = 1
Prediction = 0.9
```

The loss is small.

But:

```text
True label = 1
Prediction = 0.01
```

The loss is very large.

Therefore, the training objective is:

```text
Minimize loss
```

---

# 11. Gradient

The gradient tells us how the loss changes when the parameters change.

Conceptually:

```text
Parameter
    ↓
How does changing it affect the loss?
```

For example:

```text
∂L/∂W
```

tells us how the loss changes with respect to the weights.

Similarly:

```text
∂L/∂b
```

tells us how the loss changes with respect to the bias.

The gradient therefore gives us the direction in which the parameters should be adjusted.

---

# 12. Gradient Descent

Gradient descent updates the parameters in the direction that decreases the loss.

The basic update rule is:

```text
W ← W - η ∂L/∂W

b ← b - η ∂L/∂b
```

where:

```text
η = learning rate
```

The learning rate controls the size of each update.

Conceptually:

```text
Large learning rate
→ larger parameter updates

Small learning rate
→ smaller parameter updates
```

This will later become an experiment in the project.

---

# 13. Training Loop

A basic machine learning training loop is:

```text
Initialize parameters

Repeat:

    1. Forward pass
    2. Calculate loss
    3. Calculate gradients
    4. Update parameters
```

For Logistic Regression:

```text
X
↓
Linear
z = XW + b
↓
Sigmoid
p = sigmoid(z)
↓
Binary Cross Entropy
↓
Loss
↓
Gradient
↓
Update W and b
```

This same general structure will remain throughout the project.

---

# 14. Train / Validation / Test

A dataset is commonly divided into separate parts.

### Training set

Used to learn the parameters.

```text
Model sees training data
→ calculates loss
→ updates parameters
```

### Validation set

Used to evaluate the model during development.

It helps us compare:

* learning rates
* architectures
* hyperparameters
* training strategies

without using the final test set.

### Test set

Used for the final evaluation.

The model should not use test data to make training decisions.

The general workflow is:

```text
Training data
      ↓
   Training
      ↓
Validation data
      ↓
Model selection / tuning
      ↓
Final model
      ↓
Test data
      ↓
Final evaluation
```

---

# 15. Batch Training

Instead of processing the entire dataset at once, we can divide it into batches.

For example:

```text
Dataset = 60,000 samples

Batch size = 64

Batch 1 → 64 samples
Batch 2 → 64 samples
Batch 3 → 64 samples
...
```

The model calculates the gradient using each batch and updates its parameters.

This is called **mini-batch gradient descent**.

It will become useful when training the MNIST models.

---

# 16. From Logistic Regression to Softmax Regression

This is one of the most important transitions in the project.

Logistic Regression handles:

```text
2 classes
```

using:

```text
Linear → Sigmoid → Probability
```

Softmax Regression extends the idea to:

```text
K classes
```

using:

```text
Linear → Softmax → Class probabilities
```

For MNIST:

```text
784 input features
       ↓
Linear layer
       ↓
10 logits
       ↓
Softmax
       ↓
10 probabilities
```

For example:

```text
[0.01, 0.02, 0.80, 0.03, ..., 0.01]
```

The probabilities sum to approximately:

```text
1.0
```

and the largest probability determines the predicted class.

---

# 17. What Changes and What Stays the Same?

This modularity is an important design principle for the project.

### Logistic Regression

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

### Softmax Regression

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

The fundamental training process remains:

```text
Forward
 ↓
Loss
 ↓
Gradient
 ↓
Update
```

Only some components change.

This allows us to reuse much of the implementation.

---

# 18. What We Do NOT Need Yet

At this stage, we do not need to deeply study:

* Convolution
* Pooling
* Attention
* Transformers
* Vision Transformers
* CNN architectures
* GPU optimization
* Distributed training

Those concepts will be introduced when the corresponding modules are added.

The goal is to understand the current model completely before increasing complexity.

---

# 19. Foundation → Implementation

The concepts required for the first experiment are:

```text
Dataset
   ↓
Features + Labels
   ↓
Linear Model
   ↓
Weights + Bias
   ↓
Sigmoid
   ↓
Probability
   ↓
Binary Cross Entropy
   ↓
Gradient
   ↓
Gradient Descent
   ↓
Prediction
   ↓
Evaluation
```

The first implementation will therefore be:

```text
Simple Binary Dataset
        ↓
Logistic Regression
        ↓
NumPy from scratch
        ↓
Train
        ↓
Evaluate
        ↓
Visualize decision boundary
        ↓
Experiment with hyperparameters
        ↓
Document results
```

After that:

```text
Logistic Regression
        ↓
Generalize binary → multiclass
        ↓
Softmax Regression
        ↓
MNIST
```

---

# 20. Key Takeaways

Before starting implementation, we should be able to explain:

1. What supervised learning is.
2. What classification means.
3. The difference between binary and multiclass classification.
4. What features and labels are.
5. What weights and bias represent.
6. Why bias is needed.
7. What a linear transformation does.
8. What logits are.
9. How sigmoid converts a logit into a probability.
10. What a loss function measures.
11. What a gradient represents.
12. How gradient descent updates parameters.
13. What a training loop does.
14. The difference between training, validation, and test data.
15. Why Logistic Regression can be extended to Softmax Regression.
16. Which parts of the pipeline can be reused between models.

The most important conceptual chain is:

```text
Input
  ↓
Linear transformation
  ↓
Logits
  ↓
Probability transformation
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Parameter update
```

This chain will remain the foundation of the project even as the model becomes significantly more complex.
