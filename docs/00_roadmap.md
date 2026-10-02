# Project Roadmap

## 1. Roadmap Philosophy

This personal learning project follows a **progressive and modular approach**. Its purpose is to demonstrate understanding of Softmax Regression, MLP, CNN, and Vision Transformer, including their underlying mathematics. Supporting documents preserve the studied theory for future revision.

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
Evaluate / Analyze Results
    ↓
Move to Next Architecture
```

The notebooks contain the learning process and exploratory experiments. After the basic architecture progression through ViT is complete, `experiments/` will contain simple controlled Softmax studies and deeper MLP studies. Advanced CNN and ViT experiments are outside this repository's scope and are deferred to a future computer-vision project using more complex datasets.

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
Controlled Softmax / MLP experiments
Systematic evaluation
Training / architecture analysis
```

The notebook experiments are primarily used while learning and developing the model.

The `experiments/` folder is used later for systematic Softmax and MLP studies. CNN and ViT remain basic architecture studies, with evaluation and architectural comparison in the learning notebooks and supporting documentation.

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
Evaluation / Analysis
   ↓
Next Model
```

---
**Current order:** complete the basic architectures through ViT first, then perform the structured Softmax and MLP experiments. Existing notebook experiments remain part of the learning record.

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
Just imagine Single layer Perceptron.

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
Epochs / full-batch iterations
Batch size
Optional basic optimizer / regularization settings
Training and validation behavior
Convergence speed
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
```

## Structured Experiments

MLP is the main experimental architecture. Later studies in `experiments/` are guided by AI VIET NAM's **“Insight into Multi-layer Perceptron” by Quang-Vinh Dinh**:

```text
Data normalization
Width / depth
Activation functions
Parameter initialization
Batch Normalization
SGD vs Adam
Gradient behavior in deeper networks
```

These studies connect training choices to model behavior and document observations and conclusions.

## Deliverables

```text
notebooks/04_mlp.ipynb
docs/04_mlp.md
src/04_mlp.py
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

## Basic Learning Topics

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

## Basic Development

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
Basic comparison with MLP and Softmax
```

## Experiment Scope

The CNN baseline is complete in `notebooks/05_cnn.ipynb`, `src/05_cnn.py`, and `docs/05_cnn.md`. This stage focuses on convolution, local connectivity, weight sharing, spatial representations, and the mathematics behind the operations.

Advanced CNN architecture studies, extensive augmentation, regularization studies, and transfer learning are deferred to a future computer-vision repository with more challenging datasets.

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
Classification Token / Head
```

## Reused Components

```text
MNIST
Training Pipeline
Loss
Optimization
Evaluation
```

## Basic Deliverables

```text
notebooks/06_vit.ipynb
docs/06_vit.md
src/06_vit.py
```

Build, train, and evaluate a basic MNIST ViT, explain its mathematics and tensor representations, and compare it with the basic CNN, MLP, and Softmax models.

Advanced ViT tuning, attention variants, augmentation, pretraining, and transfer learning are deferred to a future computer-vision repository.

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

The implementation progression uses two layers:

## Layer A — From Scratch

Use NumPy for Logistic Regression and Softmax Regression to understand the underlying mathematics.

```text
Manual computation
      ↓
Manual gradients
      ↓
Manual parameter updates
```

## Layer B — PyTorch

Use PyTorch for the Softmax framework comparison and the basic MLP, CNN, and ViT implementations. Explain their mathematics in the supporting documents; separate NumPy implementations of these deeper architectures are not required.

```text
PyTorch modules
      ↓
Autograd
      ↓
Optimizer
```

Compare the NumPy and PyTorch Softmax implementations to understand what the framework automates and check their behavior.

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

Performed in the `experiments/` folder for Softmax Regression and MLP after the basic architecture progression is complete.

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

Once the basic MNIST models have been implemented, compare them under a consistent evaluation setup:

```text
Softmax Regression
MLP
CNN
Vision Transformer
```

The main comparison records:

```text
Validation / Test Accuracy
Parameter Count
Training Time (where useful)
```

Include other metrics or error analysis only when they help explain the basic models. This comparison does not require advanced CNN or ViT optimization.

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
Evaluation / Architectural Comparison
```

---

Structured experiments follow the basic architecture progression and cover only Softmax Regression and MLP. See the root README for the current progress checklist and `experiments/README.md` for the detailed experiment plan.

