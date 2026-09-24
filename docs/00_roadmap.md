# Project Roadmap

## 1. Roadmap Philosophy

This project follows a **progressive and modular approach**.

Rather than treating Logistic Regression, Softmax Regression, MLP, CNN, and Vision Transformer as completely separate projects, each stage will build on concepts and components from the previous stage.

The central question at every stage is:

> **What limitation does the current model have, and what component does the next model add, replace, or modify to address it?**

The model progression is:

```text
Foundations
    ↓
Logistic Regression
    ↓
Softmax Regression
    ↓
PyTorch Softmax Regression
    ↓
   MLP
    ↓
   CNN
    ↓
Vision Transformer
```

Each major model stage follows a common development cycle:

```text
Learn Theory
    ↓
Build Learning Notebook
    ↓
Understand / Test Model
    ↓
Extract Clean src/ Implementation
    ↓
Write Supporting Documentation
    ↓
Run Structured Experiments
    ↓
Analyze Results
    ↓
Move to Next Architecture
```

The notebooks contain the learning process and exploratory experiments, while the `experiments/` folder is used for more systematic and controlled evaluation after a model has been implemented and understood.

Logistic Regression is a short binary-classification learning stage, while **Softmax Regression on MNIST is the first main project milestone**.

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

# 4. Project Development Workflow

Each model is developed across four main repository areas.

```text
notebooks/
Step-by-step learning
Implementation
Exploratory experiments
Visualizations
Observations

        ↓

src/
Clean reusable implementation

        ↓

docs/
Deeper theory
Mathematical derivations
Architectural explanations

        ↓

experiments/
Controlled experiments
Systematic evaluation
Model comparisons
Broader analysis
```

The notebook experiments are primarily used while learning and developing the model.

The `experiments/` folder is used later to test the model more systematically and provide a more complete picture of its behavior.

A typical stage therefore follows:

```text
Question
   ↓
Theory
   ↓
Notebook Implementation
   ↓
Exploratory Testing
   ↓
Clean Source Implementation
   ↓
Documentation
   ↓
Structured Experiments
   ↓
Analysis
   ↓
Next Model
```

---

# 5. Stage 0 — Foundations

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

# 6. Stage 1 — Logistic Regression

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

# 7. Stage 2 — Softmax Regression

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

# 8. Stage 3 — MLP

## Motivation

Softmax Regression maps the flattened MNIST pixels directly to class logits using a linear model.

The MLP introduces a hidden nonlinear representation.

## Architecture

Baseline architecture:

```text
784
 ↓
Linear
 ↓
128
 ↓
ReLU
 ↓
Linear
 ↓
10 logits
```

## New Components

```text
Hidden Layers
ReLU Activation
Nonlinear Representation
Backpropagation Through Multiple Layers
Model Width
Model Depth
```

## Reused Components

```text
MNIST
Data Pipeline
Mini-batch Training
Cross Entropy
SGD
Train / Validation / Test Split
Evaluation
```

## Learning Experiments

The development notebook investigates:

```text
Hidden-layer width
Network depth
Parameter count
Training vs validation accuracy
```

Observed validation accuracy for hidden width:

```text
H = 32  → 95.82%
H = 128 → 97.11%
H = 512 → 97.50%
```

Depth experiment:

```text
1 hidden layer  → 97.11%
2 hidden layers → 97.18%
3 hidden layers → 97.10%
```

These experiments demonstrate that increasing model capacity does not necessarily produce proportional improvements in generalization.

## Deliverables

```text
notebooks/03_mlp.ipynb
docs/04_mlp.md
src/mlp.py
```

## Key Transition

```text
Softmax Regression
Linear representation
        ↓
Add hidden layer + ReLU
        ↓
MLP
Nonlinear representation
```

## Remaining Limitation

The input image is still flattened into a 784-dimensional vector.

The MLP therefore does not explicitly exploit the two-dimensional spatial structure of the image.

This motivates the transition to CNN.

---

# 9. Stage 4 — CNN

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

## Planned Learning Topics

```text
Image tensor representation
Channels
Convolution
Kernel / Filter
Local receptive field
Weight sharing
Feature maps
Stride
Padding
Pooling
Shape tracking
Hierarchical feature learning
CNN inductive bias
```

## Planned Development

```text
Theory
 ↓
CNN learning notebook
 ↓
Baseline CNN
 ↓
Evaluation
 ↓
Clean src implementation
 ↓
CNN documentation
 ↓
Structured CNN experiments
```

## Planned Experiments

Possible structured experiments include:

```text
Number of convolutional channels
Kernel size
Network depth
Pooling configuration
SGD vs Adam
Learning rate
Regularization
Data augmentation
MLP vs CNN comparison
```

Not every experiment must be included; experiments will be selected based on what is useful for understanding the model.

---

# 10. Stage 5 — Vision Transformer

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

# 11. Architecture Evolution

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

# 12. Reusable vs Replaceable Components

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

# 13. Implementation Layers

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

# 14. Development and Experiment Cycle

Two types of experiments are used in this project.

## Exploratory Experiments

Performed inside the learning notebooks while developing and understanding a model.

```text
Question
   ↓
Learn Theory
   ↓
Implement
   ↓
Small Experiment
   ↓
Observe
   ↓
Understand
```

These experiments may be informal and focus on one concept at a time.

## Structured Experiments

Performed in the `experiments/` folder after the model has been implemented and understood.

```text
Define Question
   ↓
Choose Controlled Variables
   ↓
Run Reproducible Experiment
   ↓
Record Metrics
   ↓
Compare Results
   ↓
Analyze
   ↓
Summarize Findings
```

Structured experiments are intended to provide a broader and more systematic picture of model behavior.

---

# 15. Final Model Comparison

Once the major MNIST models have been implemented and systematically evaluated, the project will compare:

```text
Softmax Regression
MLP
CNN
Vision Transformer
```

Possible comparison metrics include:

```text
Training Accuracy
Validation Accuracy
Test Accuracy
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

The comparison should also include architectural differences:

```text
Representation
Model capacity
Spatial inductive bias
Optimization behavior
Computational cost
Error patterns
```

The objective is not simply to identify the model with the highest accuracy, but to understand **what architectural change produced each improvement and what trade-offs were introduced**.

---

# 16. Final Learning Path

```text
FOUNDATIONS
    │
    ▼
Logistic Regression
Binary Classification
    │
    │ "Multiple classes?"
    ▼
Softmax Regression
MNIST
    │
    │ "What does PyTorch automate?"
    ▼
PyTorch Softmax Regression
    │
    │ "Need nonlinear representation?"
    ▼
MLP
MNIST
    │
    │ "Need spatial understanding?"
    ▼
CNN
MNIST
    │
    │ "Alternative representation to convolution?"
    ▼
Vision Transformer
MNIST
```

For each major architecture:

```text
Theory
  ↓
Learning Notebook
  ↓
Clean src Implementation
  ↓
Documentation
  ↓
Structured Experiments
  ↓
Model Comparison
```

---
