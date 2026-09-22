# MNIST Image Classification — From Scratch to Deep Learning

## Overview

This project explores image classification through a progressive learning path, starting from the mathematical foundations of machine learning and gradually moving toward modern deep learning architectures.

The project uses a small binary classification dataset to study **Logistic Regression** first. This provides the mathematical foundation needed to understand **Softmax Regression**, which is the first model applied to the main dataset: **MNIST**.

The overall progression is:

```text
Foundations
    ↓
Logistic Regression
    ↓
Softmax Regression ────── MNIST
    ↓
MLP ───────────────────── MNIST
    ↓
CNN ───────────────────── MNIST
    ↓
Vision Transformer ────── MNIST
```

The goal is not simply to achieve high accuracy. The project focuses on understanding **why each architecture works, what its limitations are, and how each new model modifies or reuses components from the previous one**.

---

## Objectives

* Understand the mathematical foundations behind classification models.
* Implement important algorithms from scratch using NumPy.
* Test and visualize Logistic Regression on a simple binary classification dataset.
* Implement Softmax Regression from scratch and apply it to MNIST.
* Reimplement models using PyTorch where appropriate.
* Understand what machine learning frameworks automate.
* Experiment with hyperparameters and training configurations.
* Analyze model errors rather than relying only on accuracy.
* Gradually extend the classifier from linear models to neural networks, CNNs, and potentially Vision Transformers.
* Maintain documentation throughout the development process.

---

## Datasets

### Binary Classification Dataset

Logistic Regression will first be developed and tested on a small binary classification dataset.

The purpose is to make concepts such as:

* Linear decision boundaries
* Sigmoid
* Probability
* Binary Cross Entropy
* Gradient Descent
* Model parameters

easy to understand and visualize.

A simple dataset will be used so that the decision boundary can be plotted directly.

### MNIST

Starting from Softmax Regression, the main dataset will be **MNIST**.

Each sample is a grayscale image of:

```text
28 × 28 pixels
```

representing one of ten digit classes:

```text
0 1 2 3 4 5 6 7 8 9
```

For Logistic/Softmax-style linear models, images will initially be flattened:

```text
28 × 28
   ↓
784-dimensional vector
```

Later architectures will use more appropriate image representations.

---

## Project Structure

```text
mnist-ai-classifier/
│
├── README.md
├── .gitignore
│
├── docs/
│   ├── 00_roadmap.md
│   ├── 01_foundation.md
│   ├── 02_logistic_regression.md
│   └── 03_softmax_regression.md
│
├── notebooks/
│
├── src/
│
└── experiments/
```

### `docs/`

Contains theoretical documentation, mathematical derivations, implementation explanations, and conclusions.

### `notebooks/`

Contains exploratory experiments, visualizations, training runs, and model development.

### `src/`

Contains cleaned and reusable implementations.

### `experiments/`

Contains experiment results, figures, comparisons, and other generated artifacts.

---

## Modular Design

The project treats a classifier as a collection of components rather than a single monolithic model.

```text
Data
 ↓
Preprocessing
 ↓
Representation
 ↓
Model
 ↓
Activation / Output
 ↓
Loss
 ↓
Optimization
 ↓
Evaluation
```

Different architectures will replace, add, or reuse these components.

For example:

### Logistic Regression

```text
Data
 ↓
Preprocessing
 ↓
Flatten
 ↓
Linear
 ↓
Sigmoid
 ↓
Binary Cross Entropy
 ↓
Gradient Descent
```

### Softmax Regression

```text
Data
 ↓
Preprocessing
 ↓
Flatten
 ↓
Linear
 ↓
Softmax
 ↓
Cross Entropy
 ↓
Gradient Descent
```

### MLP

```text
Data
 ↓
Preprocessing
 ↓
Flatten
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Cross Entropy
 ↓
Optimizer
```

### CNN

```text
Data
 ↓
Image Representation
 ↓
Convolution
 ↓
Activation
 ↓
Pooling
 ↓
Convolution
 ↓
Linear
 ↓
Cross Entropy
 ↓
Optimizer
```

### Vision Transformer

```text
Data
 ↓
Image
 ↓
Patch Embedding
 ↓
Positional Encoding
 ↓
Transformer Encoder
 ↓
Classification Head
 ↓
Cross Entropy
 ↓
Optimizer
```

---

## Learning & Development Workflow

Each major stage follows the same process:

```text
Question
   ↓
Theory
   ↓
Mathematical Formulation
   ↓
From-Scratch Implementation
   ↓
Experiment
   ↓
Analysis
   ↓
Framework Implementation
   ↓
Comparison
   ↓
Documentation
```

The implementation will be developed incrementally rather than starting with a complete high-level framework.

---

## Model Progression

| Stage | Model               | Dataset        | Main Purpose                                          |
| ----- | ------------------- | -------------- | ----------------------------------------------------- |
| 0     | Foundations         | —              | Mathematics and ML fundamentals                       |
| 1     | Logistic Regression | Binary dataset | Understand linear binary classification               |
| 2     | Softmax Regression  | MNIST          | Multiclass linear classification                      |
| 3     | MLP                 | MNIST          | Introduce nonlinear representations                   |
| 4     | CNN                 | MNIST          | Exploit spatial image structure                       |
| 5     | Vision Transformer  | MNIST          | Explore patch-based representation and self-attention |

The initial project milestone is **Softmax Regression on MNIST**, as specified by the project requirements.

---

## Planned Experiments

Experiments may investigate:

* Learning rate
* Number of epochs
* Batch size
* Weight initialization
* Training and validation behavior
* Convergence
* Model capacity
* Generalization
* Error patterns

Evaluation will include:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy
* Test accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Later models may additionally be compared using:

* Number of parameters
* Training time
* Inference time
* Memory usage
* Computational requirements

---

## Documentation Strategy

Each major module will document:

1. Motivation
2. Problem definition
3. Theory
4. Mathematical formulation
5. Implementation
6. Experiments
7. Results
8. Error analysis
9. Limitations
10. What changed from the previous model?
11. What was reused?
12. What comes next?

The documentation should explain not only **what** was implemented, but also **why the architecture evolves**.

---

## Implementation Strategy

Where appropriate, models will be implemented at two levels.

### From Scratch

NumPy will be used to explicitly implement important mathematical operations.

Examples:

```text
Linear transformation
Sigmoid
Softmax
Cross Entropy
Gradient computation
Gradient Descent
Activation functions
```

### Framework Implementation

PyTorch will be introduced after the underlying concepts have been implemented and understood.

The goal is to compare the manual implementation with the framework implementation and understand what PyTorch automates.

---

## Current Progress

### Project Setup

* [x] Create GitHub repository
* [x] Create project folder structure
* [x] Create initial README
* [x] Create roadmap
* [x] Set up first development notebook

### Foundations

* [x] Review required linear algebra
* [x] Review supervised learning
* [x] Understand parameters and bias
* [x] Understand loss functions
* [x] Understand gradient descent
* [x] Understand train / validation / test

### Logistic Regression

* [x] Theory
* [x] Sigmoid
* [x] Binary Cross Entropy
* [x] Gradient derivation
* [x] NumPy implementation
* [x] Train on binary dataset
* [x] Visualize decision boundary
* [x] Evaluate model
* [x] Document results

### Softmax Regression — Basic Implementation

* [x] Multiclass classification
* [x] Softmax
* [x] Numerical stability
* [x] Cross Entropy
* [x] Gradient derivation
* [x] NumPy implementation
* [x] Train on MNIST
* [x] Evaluate model
* [x] PyTorch implementation
* [x] NumPy vs PyTorch comparison

### Softmax Regression — Optimization & Experiments

* [ ] Establish a fixed baseline configuration
* [ ] Compare SGD, SGD + Momentum, Nesterov, Adam, and AdamW
* [ ] Experiment with learning rate
* [ ] Experiment with batch size
* [ ] Experiment with number of epochs
* [ ] Add weight decay / L2 regularization
* [ ] Add learning-rate scheduling
* [ ] Add early stopping and best-model checkpointing
* [ ] Experiment with label smoothing
* [ ] Perform confusion-matrix and misclassification analysis
* [ ] Compare convergence speed and final validation/test performance
* [ ] Document optimization experiments and conclusions
* [ ] Refactor final reusable Softmax code into `src/`

### MLP — Basic Implementation

* [x] Build a basic MLP for MNIST
* [x] Add hidden layer(s) and nonlinear activation
* [x] Train and evaluate the basic model

### MLP — Optimization & Experiments

* [ ] Establish a reproducible MLP baseline
* [ ] Compare hidden-layer sizes and number of layers
* [ ] Compare ReLU and GELU activations
* [ ] Compare SGD, Adam, and AdamW
* [ ] Tune learning rate and batch size
* [ ] Add weight decay
* [ ] Add Dropout
* [ ] Experiment with Batch Normalization
* [ ] Add learning-rate scheduling
* [ ] Add early stopping and checkpointing
* [ ] Analyze overfitting and generalization
* [ ] Compare optimized MLP against Softmax Regression
* [ ] Document MLP theory, experiments, and conclusions
* [ ] Refactor final reusable MLP code into `src/`

### CNN — Basic Implementation

* [ ] Study convolution, kernels, feature maps, stride, padding, and pooling
* [ ] Build a basic CNN in PyTorch
* [ ] Keep the original 2D image representation
* [ ] Train the CNN on MNIST
* [ ] Evaluate accuracy and confusion matrix
* [ ] Compare CNN against MLP and Softmax Regression
* [ ] Document the basic CNN architecture
* [ ] Refactor basic CNN components into `src/`

### CNN — Optimization & Experiments

* [ ] Tune number of convolutional layers and channels
* [ ] Experiment with kernel size, stride, padding, and pooling
* [ ] Add Batch Normalization
* [ ] Add Dropout where appropriate
* [ ] Compare Adam and AdamW
* [ ] Tune learning rate and weight decay
* [ ] Add learning-rate scheduling
* [ ] Experiment with image normalization and data augmentation
* [ ] Add early stopping and checkpointing
* [ ] Analyze difficult and misclassified samples
* [ ] Compare parameter count, training time, and test performance
* [ ] Document optimized CNN experiments and conclusions
* [ ] Refactor final optimized CNN into `src/`

### Vision Transformer — Basic Implementation

* [ ] Study patch embedding
* [ ] Study positional encoding / positional embeddings
* [ ] Study self-attention and multi-head attention
* [ ] Study Transformer encoder blocks
* [ ] Study the classification token / classification head
* [ ] Build a basic ViT for MNIST in PyTorch
* [ ] Train and evaluate the basic ViT
* [ ] Compare ViT against CNN, MLP, and Softmax Regression
* [ ] Document the basic ViT architecture
* [ ] Refactor basic ViT components into `src/`

### Vision Transformer — Optimization & Experiments

* [ ] Tune patch size
* [ ] Tune embedding dimension
* [ ] Tune number of attention heads
* [ ] Tune number of Transformer encoder blocks
* [ ] Tune MLP expansion ratio inside Transformer blocks
* [ ] Add Dropout / attention dropout
* [ ] Use AdamW with weight decay
* [ ] Add learning-rate warmup and scheduling
* [ ] Experiment with normalization and augmentation
* [ ] Add early stopping and checkpointing
* [ ] Analyze attention behavior where useful
* [ ] Compare parameter count, training time, and final performance
* [ ] Document optimized ViT experiments and conclusions
* [ ] Refactor final optimized ViT into `src/`

### Final Model Comparison

* [ ] Compare Softmax Regression, MLP, CNN, and ViT under a consistent evaluation setup
* [ ] Compare validation/test accuracy
* [ ] Compare precision, recall, F1-score, and confusion matrices
* [ ] Compare convergence behavior
* [ ] Compare parameter counts
* [ ] Compare training and inference time
* [ ] Summarize strengths, limitations, and architectural differences
* [ ] Produce final project conclusions
