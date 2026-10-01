# Experiments

This folder contains systematic experiments performed after the basic model implementations are understood.

The purpose is not simply to maximize MNIST accuracy. Each experiment should investigate a specific training or modeling choice and observe how it affects model behavior.

The experiment scope of this repository is:

```text
Softmax Regression
        ↓
Simple Hyperparameter Experiments
        ↓
MLP
        ↓
Deeper Training / Architecture Experiments
```

CNN and Vision Transformer are **not included in the advanced experiment scope of this repository**.

They are implemented on MNIST to understand their basic architectures, but deeper CNN and ViT experimentation will be left for a future project using more complex datasets.

---

# 1. Softmax Regression

Softmax Regression will be used for simple hyperparameter experiments.

The goal is to understand how basic training choices affect a simple linear classifier before studying deeper neural networks.

Possible experiments include:

```text
Learning rate
Batch size
Number of epochs
Optimizer
Regularization
```

Useful measurements:

```text
Training loss
Validation loss
Training accuracy
Validation accuracy
Convergence speed
```

These experiments should remain simple.

The purpose is mainly to build intuition about the training process before moving to MLPs.

---

# 2. Multi-Layer Perceptron

MLP will be the main experimental architecture in this repository.

The experiments will be guided primarily by:

```text
AI VIET NAM – AI Course 2025
"Insight into Multi-layer Perceptron"
Quang-Vinh Dinh
```

The document studies the major decisions involved in training an MLP:

```text
Data normalization
        ↓
Network construction
        ↓
Parameter initialization
        ↓
Optimizer selection
        ↓
Loss function selection
        ↓
Metric selection
```

The MLP experiments will therefore investigate these components progressively rather than treating them as unrelated optimization tricks.

---

## 2.1 Data Normalization

Compare different input normalization strategies such as:

```text
[0, 1]
[-1, 1]
Z-score normalization
```

Observe how normalization affects:

```text
Training behavior
Convergence
Training accuracy
Validation accuracy
```

---

## 2.2 Hidden Layer Width

Investigate how the number of neurons in a hidden layer affects the model.

For example:

```text
Small hidden layer
        vs
Medium hidden layer
        vs
Large hidden layer
```

Observe:

```text
Parameter count
Training accuracy
Validation accuracy
Training time
```

The purpose is to understand how increasing model capacity affects an MLP.

---

## 2.3 Network Depth

Compare MLPs with different numbers of hidden layers.

```text
1 hidden layer
        ↓
2 hidden layers
        ↓
3 hidden layers
        ↓
deeper MLP
```

Observe how increasing depth changes:

```text
Training behavior
Accuracy
Gradient behavior
Optimization difficulty
```

This also provides the foundation for studying problems that appear when neural networks become deeper.

---

## 2.4 Activation Functions

Compare different activation functions covered in the reference material.

Examples include:

```text
Sigmoid
Tanh
ReLU
Leaky ReLU
ELU
PReLU
Swish
GELU
```

Not every activation must necessarily receive a full experiment.

The goal is to understand:

```text
Why nonlinear activations are needed
        ↓
How different activations behave
        ↓
How activation choice affects training
```

Particular attention can be given to problems such as:

```text
Sigmoid saturation
Dying ReLU
```

---

## 2.5 Parameter Initialization

Investigate how initialization affects neural-network training.

Possible comparisons include:

```text
Simple / default initialization
        vs
Xavier initialization
        vs
He / Kaiming initialization
```

Observe:

```text
Activation scale
Gradient behavior
Training stability
Convergence
```

Initialization experiments should be connected to the activation functions used by the network.

---

## 2.6 Batch Normalization

Investigate the effect of adding Batch Normalization to deeper MLPs.

Compare:

```text
Without BatchNorm
        vs
With BatchNorm
```

Observe:

```text
Training stability
Convergence speed
Activation behavior
Validation performance
```

---

## 2.7 Optimizer Selection

Compare basic optimizers such as:

```text
SGD
 vs
Adam
```

Observe:

```text
Training loss
Convergence speed
Validation performance
Training stability
```

The purpose is to understand how the parameter-update strategy affects training rather than simply selecting the optimizer that reaches the highest accuracy.

---

## 2.8 Gradient Behavior

As the MLP becomes deeper, directly inspect gradient flow through the network.

For example:

```text
Layer 1 → gradient norm
Layer 2 → gradient norm
Layer 3 → gradient norm
...
Layer N → gradient norm
```

This can help demonstrate problems such as:

```text
Vanishing gradients
Exploding gradients
```

and connect those problems to earlier experiments involving:

```text
Activation functions
Initialization
Normalization
Network depth
```

---

# Experiment Philosophy

Each experiment should answer a small question.

```text
What are we changing?
        ↓
Why might it matter?
        ↓
What should we measure?
        ↓
What actually changes?
        ↓
What did we learn?
```

Where possible, change **one main variable at a time**.

The goal is not:

```text
"Which setup gives the highest MNIST accuracy?"
```

The goal is:

```text
"Why does this design or training choice change
the behavior of the model?"
```

---

# CNN and Vision Transformer

CNN and Vision Transformer are still part of the main learning progression:

```text
Softmax
   ↓
MLP
   ↓
CNN
   ↓
Vision Transformer
```

However, MNIST is being used primarily to make these architectures easy to understand and implement.

For this repository:

```text
CNN
→ learn convolution, local connectivity,
  weight sharing, and spatial feature extraction

ViT
→ learn patches, embeddings,
  self-attention, and Transformer encoders
```

Advanced experiments involving CNN and Vision Transformer are intentionally **out of scope**.

Topics such as:

```text
Advanced CNN architectures
ResNet studies
Data augmentation strategies
Transfer learning
Advanced ViT variants
Attention optimization
Modern image-training techniques
Large architecture comparisons
```

will be better explored in a separate, more advanced computer-vision repository using more challenging datasets.

---

# Current Plan

```text
Basic Architectures

Softmax ✓
   ↓
MLP ✓
   ↓
CNN ✓
   ↓
Vision Transformer
```

Then:

```text
Experiments

Softmax
   ↓
Simple hyperparameter study

MLP
   ↓
Normalization
   ↓
Width / depth
   ↓
Activation functions
   ↓
Initialization
   ↓
Batch Normalization
   ↓
Optimizers
   ↓
Gradient behavior
```

And later, in a separate project:

```text
Advanced Computer Vision Repository

More complex datasets
        ↓
Advanced CNN experiments
        ↓
Advanced Vision Transformer experiments
```

This keeps the current MNIST repository focused on **learning the progression of architectures and understanding fundamental neural-network training behavior**, without forcing advanced computer-vision experiments onto a dataset that is too simple for that purpose.
