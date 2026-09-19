# Project Roadmap

## 1. Roadmap Philosophy

This project follows a **progressive and modular approach**.

Rather than treating Logistic Regression, Softmax Regression, MLP, CNN, and Vision Transformer as completely separate projects, each stage will build on concepts and components from the previous stage.

The central question at every stage is:

> **What limitation does the current model have, and what component does the next model add, replace, or modify to address it?**

The progression is:

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

Logistic Regression is a short binary-classification learning stage, while **Softmax Regression on MNIST is the main initial project milestone**.

---

# 2. High-Level Roadmap

```text
┌───────────────────────────────┐
│       0. FOUNDATIONS          │
│                               │
│ Linear Algebra                │
│ Supervised Learning           │
│ Loss Functions                │
│ Gradient Descent              │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│    1. LOGISTIC REGRESSION     │
│                               │
│ Binary Dataset                │
│ Linear Model                  │
│ Sigmoid                       │
│ Binary Cross Entropy          │
│ Gradient Descent              │
└───────────────┬───────────────┘
                │
                │ Extend binary
                │ → multiclass
                ▼
┌───────────────────────────────┐
│     2. SOFTMAX REGRESSION     │
│                               │
│ MNIST                         │
│ Linear Model                  │
│ Softmax                       │
│ Cross Entropy                 │
│ Gradient Descent              │
└───────────────┬───────────────┘
                │
                │ Add nonlinear
                │ representation
                ▼
┌───────────────────────────────┐
│             3. MLP            │
│                               │
│ MNIST                         │
│ Hidden Layers                 │
│ Activation Functions          │
│ Backpropagation               │
└───────────────┬───────────────┘
                │
                │ Introduce spatial
                │ representation
                ▼
┌───────────────────────────────┐
│             4. CNN            │
│                               │
│ MNIST                         │
│ Convolution                   │
│ Feature Maps                  │
│ Pooling                       │
│ Spatial Features              │
└───────────────┬───────────────┘
                │
                │ Introduce token-based
                │ representation
                ▼
┌───────────────────────────────┐
│       5. VISION TRANSFORMER   │
│                               │
│ MNIST                         │
│ Image Patches                 │
│ Patch Embeddings              │
│ Positional Encoding           │
│ Self-Attention                │
│ Transformer Encoder           │
└───────────────────────────────┘
```

---

# 3. Modular Architecture

The classifier is divided conceptually into reusable modules:

```text
┌──────────────┐
│     DATA     │
└──────┬───────┘
       ↓
┌──────────────┐
│PREPROCESSING │
└──────┬───────┘
       ↓
┌──────────────┐
│REPRESENTATION│
└──────┬───────┘
       ↓
┌──────────────┐
│    MODEL     │
└──────┬───────┘
       ↓
┌──────────────┐
│    OUTPUT    │
└──────┬───────┘
       ↓
┌──────────────┐
│     LOSS     │
└──────┬───────┘
       ↓
┌──────────────┐
│ OPTIMIZATION │
└──────┬───────┘
       ↓
┌──────────────┐
│  EVALUATION  │
└──────────────┘
```

The purpose of this design is to make components easy to:

* Reuse
* Replace
* Remove
* Extend
* Compare

---

# 4. Stage 0 — Foundations

## Goal

Understand the minimum mathematics and machine learning concepts required to implement the first classifier.

## Topics

### Linear Algebra

```text
Vectors
Matrices
Shapes
Transpose
Dot Product
Matrix Multiplication
```

### Machine Learning

```text
Features
Labels
Parameters
Bias
Prediction
Loss
Training
Validation
Testing
```

### Optimization

```text
Gradient
Gradient Descent
Learning Rate
Epoch
Batch
```

## Deliverable

A short foundation document explaining the concepts required by the following stages.

---

# 5. Stage 1 — Logistic Regression

## Dataset

A small binary classification dataset.

The dataset should preferably contain only a few features so that the classifier's decision boundary can be visualized.

## Architecture

```text
Input
  ↓
Linear Transformation
  ↓
Sigmoid
  ↓
Probability
  ↓
Binary Cross Entropy
```

Mathematically:

```text
z = Wx + b

p = sigmoid(z)
```

## Concepts

```text
Linear classifier
Parameters
Bias
Sigmoid
Probability
Binary Cross Entropy
Gradient
Gradient Descent
Decision Boundary
```

## Implementation

First implement the model using NumPy.

## Experiments

At minimum:

```text
Train the classifier
Plot the decision boundary
Plot training loss
Evaluate predictions
Experiment with learning rate
```

## Purpose

Logistic Regression serves as the simplest complete classification system and establishes the concepts that will be reused in Softmax Regression.

---

# 6. Stage 2 — Softmax Regression

## Dataset

MNIST.

```text
28 × 28 grayscale image
        ↓
784-dimensional vector
        ↓
10 classes
```

## Architecture

```text
Input
  ↓
Flatten
  ↓
Linear Transformation
  ↓
Softmax
  ↓
10 Class Probabilities
  ↓
Cross Entropy
```

## Concepts

```text
Multiclass classification
Logits
Softmax
Numerical stability
Cross Entropy
Gradient derivation
Batch training
```

## Key Transition

Logistic Regression:

```text
Binary classification
      ↓
Sigmoid
      ↓
P(y = 1)
```

Softmax Regression:

```text
Multiclass classification
      ↓
Softmax
      ↓
P(y = 0...9)
```

## Implementation

```text
NumPy
  ↓
From-scratch Softmax Regression
  ↓
MNIST
```

Then:

```text
PyTorch
  ↓
Framework implementation
  ↓
Compare with NumPy version
```

## Experiments

```text
Learning rate
Epochs
Batch size
Initialization
Training loss
Validation loss
Accuracy
Confusion matrix
Error analysis
```

## Main Milestone

Successfully implement and analyze a Softmax Regression classifier on MNIST.

---

# 7. Stage 3 — MLP

## Motivation

Softmax Regression is a linear model.

Its structure is essentially:

```text
Input
  ↓
Linear
  ↓
Output
```

MLP introduces nonlinear hidden representations.

## Architecture

```text
Input
  ↓
Linear
  ↓
ReLU
  ↓
Linear
  ↓
Output
```

Example:

```text
784
 ↓
128
 ↓
64
 ↓
10
```

## New Components

```text
Hidden Layers
Activation Functions
Multiple Linear Layers
Nonlinear Representation
Backpropagation Through Multiple Layers
```

## Reused Components

```text
MNIST
Data Pipeline
Loss
Optimization
Evaluation
```

---

# 8. Stage 4 — CNN

## Motivation

MLP processes the flattened image as a vector and does not explicitly exploit the spatial structure of an image.

CNN introduces spatial feature extraction.

## Architecture

```text
Image
  ↓
Convolution
  ↓
Activation
  ↓
Pooling
  ↓
Convolution
  ↓
Activation
  ↓
Pooling
  ↓
Flatten
  ↓
Linear
  ↓
Output
```

## New Components

```text
Convolution
Kernel / Filter
Stride
Padding
Feature Maps
Pooling
Receptive Field
```

## Reused Components

```text
MNIST
Training Pipeline
Loss
Optimizer
Evaluation
```

---

# 9. Stage 5 — Vision Transformer

## Motivation

CNNs introduce spatial inductive biases through convolution.

Vision Transformers use a different representation: images are converted into sequences of patches and processed using self-attention.

## Architecture

```text
Image
  ↓
Patch Extraction
  ↓
Patch Embedding
  ↓
Positional Encoding
  ↓
Transformer Encoder
  ↓
Classification Head
  ↓
Output
```

## New Components

```text
Image Patches
Patch Embeddings
Positional Encoding
Query / Key / Value
Self-Attention
Multi-Head Attention
Transformer Encoder
```

## Reused Components

```text
MNIST
Training Pipeline
Loss
Optimization
Evaluation
```

---

# 10. Architecture Evolution

The complete evolution can be visualized as:

```text
LOGISTIC REGRESSION

Input
 ↓
Linear
 ↓
Sigmoid
 ↓
Binary Output
```

```text
        ↓
      EXTEND
        ↓
```

```text
SOFTMAX REGRESSION

Input
 ↓
Linear
 ↓
Softmax
 ↓
Multiclass Output
```

```text
        ↓
       ADD
   hidden nonlinear
   representation
        ↓
```

```text
MLP

Input
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Multiclass Output
```

```text
        ↓
     REPLACE
   representation
        ↓
```

```text
CNN

Image
 ↓
Convolution
 ↓
ReLU
 ↓
Pooling
 ↓
...
 ↓
Classifier
```

```text
        ↓
     REPLACE
   representation
        ↓
```

```text
VISION TRANSFORMER

Image
 ↓
Patches
 ↓
Embeddings
 ↓
Self-Attention
 ↓
Transformer
 ↓
Classifier
```

---

# 11. Reusable vs Replaceable Components

| Component                   | Logistic | Softmax |      MLP |      CNN |      ViT |
| --------------------------- | -------: | ------: | -------: | -------: | -------: |
| Data pipeline               |        ✓ |       ✓ |        ✓ |        ✓ |        ✓ |
| Normalization               |        ✓ |       ✓ |        ✓ |        ✓ |        ✓ |
| Flattening                  |        ✓ |       ✓ |        ✓ |        — |        — |
| Linear layer                |        ✓ |       ✓ |        ✓ |        ✓ |        ✓ |
| Sigmoid                     |        ✓ |       — |        — |        — |        — |
| Softmax                     |        — |       ✓ | optional | optional | optional |
| ReLU                        |        — |       — |        ✓ |        ✓ |        — |
| Hidden layers               |        — |       — |        ✓ |        ✓ |        ✓ |
| Convolution                 |        — |       — |        — |        ✓ |        — |
| Pooling                     |        — |       — |        — |        ✓ |        — |
| Patch embedding             |        — |       — |        — |        — |        ✓ |
| Self-attention              |        — |       — |        — |        — |        ✓ |
| Cross Entropy               |        — |       ✓ |        ✓ |        ✓ |        ✓ |
| Gradient-based optimization |        ✓ |       ✓ |        ✓ |        ✓ |        ✓ |
| Evaluation                  |        ✓ |       ✓ |        ✓ |        ✓ |        ✓ |

This table will be updated as the implementation becomes more precise.

---

# 12. Implementation Layers

Each architecture should be developed in two layers where appropriate.

## Layer A — From Scratch

Use NumPy to understand the underlying mathematics.

```text
Manual computation
      ↓
Manual gradients
      ↓
Manual parameter updates
```

## Layer B — PyTorch

Use PyTorch to build a practical implementation.

```text
PyTorch modules
      ↓
Autograd
      ↓
Optimizer
```

The two implementations should be compared to verify that they behave consistently.

---

# 13. Experiment → Documentation Cycle

For every important concept:

```text
Question
   ↓
Learn theory
   ↓
Implement
   ↓
Run experiment
   ↓
Observe
   ↓
Explain result
   ↓
Document
```

Examples:

```text
What happens if learning rate is too large?

What happens if learning rate is too small?

Why does Softmax need numerical stabilization?

What kinds of digits does Softmax Regression confuse?

Why does adding a hidden layer help?

Why does CNN exploit image structure better?

What does self-attention add compared with convolution?
```

---

# 14. Final Comparison

Once multiple models have been implemented on MNIST, they can be compared using:

```text
Accuracy
Precision
Recall
F1
Confusion Matrix
Training Loss
Validation Loss
Parameter Count
Training Time
Inference Time
Memory Usage
```

The comparison should focus on understanding **architectural differences and trade-offs**, rather than simply declaring one model universally better.

---

# 15. Final Learning Path

```text
                FUNDAMENTALS
                     │
                     ▼
            Logistic Regression
             Binary Dataset
                     │
                     │
              "Multiple classes?"
                     │
                     ▼
             Softmax Regression
                   MNIST
                     │
                     │
             "Need nonlinear
              representation?"
                     │
                     ▼
                    MLP
                   MNIST
                     │
                     │
             "Need spatial
              understanding?"
                     │
                     ▼
                    CNN
                   MNIST
                     │
                     │
             "Alternative to
              convolution?"
                     │
                     ▼
                    ViT
                   MNIST
```

## Primary Milestone

The first required milestone is:

```text
Softmax Regression
       ↓
From scratch with NumPy
       ↓
MNIST
       ↓
Training
       ↓
Evaluation
       ↓
Error Analysis
       ↓
PyTorch implementation
       ↓
Documentation
```

The later MLP, CNN, and Vision Transformer stages are extensions of the same modular project.
