# MNIST Image Classification — From Scratch to Deep Learning

## Overview

This project explores image classification on the MNIST handwritten digit dataset, progressing from fundamental machine learning methods to modern deep learning architectures.

The main goal is not only to achieve good classification accuracy, but to understand the mathematical foundations, implementation details, and practical behavior of each model.

The project follows a progressive learning approach:

```text
Mathematical & ML Foundations
            ↓
    Logistic Regression
            ↓
     Softmax Regression
            ↓
           MLP
            ↓
           CNN
            ↓
       Transformer / ViT
```

CNN and Transformer/ViT are optional extensions after completing the Softmax Regression classifier.

---

## Objectives

* Understand the mathematical foundations behind classification models.
* Implement core algorithms from scratch using NumPy.
* Reimplement the same models using PyTorch.
* Compare implementations and understand what machine learning frameworks automate.
* Experiment with hyperparameters and training configurations.
* Analyze model errors rather than relying only on accuracy.
* Document the theory, implementation, experiments, and conclusions throughout the project.

---

## Dataset

The project uses the **MNIST handwritten digit dataset**.

Each sample is a grayscale image of:

```text
28 × 28 pixels
```

representing one of ten digit classes:

```text
0 1 2 3 4 5 6 7 8 9
```

For Softmax Regression, each image will initially be flattened into a 784-dimensional feature vector.

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
│   └── 01_softmax_regression.ipynb
│
├── src/
│
└── experiments/
```

### `docs/`

Contains theoretical documentation and mathematical derivations.

### `notebooks/`

Contains exploratory work, experiments, visualizations, and model development.

### `src/`

Contains cleaned and reusable implementations.

### `experiments/`

Contains experiment results, figures, and comparisons.

---

## Learning & Development Workflow

Each model will follow the same workflow:

```text
Question
   ↓
Theory
   ↓
Mathematical formulation
   ↓
From-scratch implementation
   ↓
Experiment
   ↓
Analysis
   ↓
PyTorch implementation
   ↓
Comparison
   ↓
Documentation
```

This structure is intended to keep the project focused on understanding the models rather than simply training pre-built architectures.

---

## Models

| Stage | Model               | Implementation  |
| ----- | ------------------- | --------------- |
| 1     | Logistic Regression | NumPy           |
| 2     | Softmax Regression  | NumPy + PyTorch |
| 3     | MLP                 | NumPy/PyTorch   |
| 4     | CNN                 | PyTorch         |
| 5     | Vision Transformer  | PyTorch         |

The initial milestone is **Softmax Regression**.

---

## Planned Experiments

Experiments will investigate factors such as:

* Learning rate
* Number of epochs
* Weight initialization
* Batch size
* Training/validation behavior
* Model errors
* Confusion matrix
* Classification metrics

Later models will additionally be compared in terms of:

* Accuracy
* Number of parameters
* Training time
* Computational requirements
* Ability to exploit spatial information
* Generalization behavior

---

## Documentation

Each major topic will document:

1. Motivation
2. Problem definition
3. Mathematical intuition
4. Mathematical formulation
5. Implementation
6. Experiments
7. Results
8. Error analysis
9. Limitations
10. What was learned
11. Next step

---

## Current Progress

### Foundations

* [ ] Linear algebra review
* [ ] Supervised learning fundamentals
* [ ] Loss functions
* [ ] Gradient descent

### Softmax Regression

* [ ] Understand logistic regression
* [ ] Derive Softmax
* [ ] Understand cross-entropy
* [ ] Derive gradients
* [ ] NumPy implementation
* [ ] Train on MNIST
* [ ] Evaluate model
* [ ] Error analysis
* [ ] PyTorch implementation
* [ ] Compare NumPy and PyTorch implementations

### Deep Learning Extensions

* [ ] MLP
* [ ] CNN
* [ ] Transformer / ViT

---

## Environment

Initial development is performed using:

* Python
* Google Colab
* NumPy
* Matplotlib
* PyTorch
* GitHub

Additional libraries will be introduced only when needed.
