# MNIST Image Classification — From Scratch to Deep Learning

This repository is a progressive study of image classification, starting from simple linear models and moving toward modern deep learning architectures.

The main goal is not only to achieve higher accuracy, but to understand:

- how each model works,
- what limitation motivates the next model,
- which parts of the pipeline are reused,
- and what changes as the architecture becomes more powerful.

The learning path is:

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

Softmax Regression, MLP, CNN, and Vision Transformer are developed primarily on **MNIST**.

## Repository Structure

```text
mnist-ai-classifier/
│
├── README.md
├── docs/
├── notebooks/
├── src/
└── experiments/
```

- `docs/` — deeper theory, mathematical derivations, roadmap, and architectural explanations.
- `notebooks/` — step-by-step learning, implementation, exploratory experiments, visualizations, and observations.
- `src/` — clean reusable implementations extracted from the notebooks.
- `experiments/` — more systematic experiments and model comparisons performed after the models are understood.

For the full project plan, architecture progression, milestones, and development workflow, see:

```text
docs/00_roadmap.md
```


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
