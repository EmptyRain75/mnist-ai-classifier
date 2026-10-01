# MNIST Image Classification — From Scratch to Deep Learning

This repository is a progressive study of image classification, starting from simple linear models and moving toward modern deep learning architectures.

The main goal is not only to achieve higher accuracy, but to understand:

- how each model works,
- what limitation motivates the next model,
- which parts of the pipeline are reused,
- how the data representation changes,
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

## Experiment Scope

The basic architecture progression continues through CNN and Vision Transformer, but the systematic experiments in this repository stop at the **MLP stage**.

```text
Softmax Regression
    ↓
Simple hyperparameter experiments

MLP
    ↓
Deeper training / architecture experiments

CNN
    ↓
Basic architecture study only

Vision Transformer
    ↓
Basic architecture study only
```

Softmax Regression is used for small, controlled hyperparameter studies.

MLP is the main experimental model. Its experiments are guided by the AI VIET NAM course material **“Insight into Multi-layer Perceptron” by Quang-Vinh Dinh**, including topics such as normalization, network width/depth, activation functions, parameter initialization, Batch Normalization, optimizer choice, and gradient behavior.

Advanced CNN and Vision Transformer experiments are intentionally left for a future computer-vision repository using more complex datasets, where deeper architecture comparisons and modern training techniques are more meaningful.

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
- `experiments/` — systematic experiments for Softmax Regression and MLP after the basic models are understood.

For the full project plan, architecture progression, milestones, and development workflow, see:

```text
docs/00_roadmap.md
```

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

### Softmax Regression — Simple Experiments

The Softmax experiment stage is intentionally small and focuses on basic training hyperparameters.

* [ ] Establish a fixed baseline configuration
* [ ] Experiment with learning rate
* [ ] Experiment with batch size
* [ ] Experiment with number of epochs
* [ ] Optionally compare a small number of basic optimizer / regularization settings
* [ ] Compare training and validation behavior
* [ ] Document observations and conclusions

The goal is to build intuition about optimization on a simple linear classifier before moving to deeper-network experiments.

### MLP — Basic Implementation

* [x] Build a basic MLP for MNIST
* [x] Add hidden layer(s) and nonlinear activation
* [x] Train and evaluate the basic model
* [x] Compare hidden-layer widths
* [x] Compare different network depths
* [x] Compare the MLP against Softmax Regression

### MLP — Experiments

The MLP is the main experimental architecture in this repository.

The experiment plan is guided by the reference:

```text
AI VIET NAM – AI Course 2025
"Insight into Multi-layer Perceptron"
Quang-Vinh Dinh
```

Planned topics include:

* [ ] Compare data-normalization strategies
* [ ] Study hidden-layer width
* [ ] Study network depth
* [ ] Compare activation functions
* [ ] Demonstrate activation-related problems such as dying ReLU where useful
* [ ] Compare parameter-initialization strategies
* [ ] Experiment with Batch Normalization
* [ ] Compare basic optimizers such as SGD and Adam
* [ ] Monitor gradient behavior in deeper MLPs
* [ ] Relate normalization, activation, initialization, depth, and gradient flow
* [ ] Document experiments and conclusions

The purpose is not simply to find the highest MNIST accuracy, but to understand how important design and training choices affect the behavior of a deeper neural network.

### CNN — Basic Implementation

* [x] Study convolution, kernels, feature maps, stride, padding, and pooling
* [x] Build a basic CNN in PyTorch
* [x] Keep the original 2D image representation
* [x] Train the CNN on MNIST
* [x] Evaluate the basic model
* [x] Compare CNN against MLP and Softmax Regression
* [x] Document the basic CNN architecture
* [x] Refactor basic CNN components into `src/`

### CNN — Advanced Experiments

Advanced CNN experimentation is **outside the scope of this MNIST repository**.

This repository uses the CNN stage to understand:

```text
Local connectivity
Weight sharing
Convolution
Pooling
Spatial feature maps
Hierarchical feature extraction
```

Topics such as deeper CNN architecture studies, ResNet-style models, extensive augmentation, transfer learning, and modern CNN training strategies will be moved to a future computer-vision repository using more challenging datasets.

### Vision Transformer — Basic Implementation

* [ ] Study image patches
* [ ] Study patch embeddings
* [ ] Study positional embeddings
* [ ] Study self-attention and multi-head attention
* [ ] Study Transformer encoder blocks
* [ ] Study the classification token / classification head
* [ ] Build a basic ViT for MNIST in PyTorch
* [ ] Train and evaluate the basic ViT
* [ ] Compare ViT against CNN, MLP, and Softmax Regression
* [ ] Document the basic ViT architecture
* [ ] Refactor basic ViT components into `src/`

### Vision Transformer — Advanced Experiments

Advanced Vision Transformer experimentation is **outside the scope of this MNIST repository**.

The ViT stage is used to understand the architectural transition:

```text
Image
    ↓
Patches
    ↓
Patch embeddings
    ↓
Positional information
    ↓
Self-attention
    ↓
Transformer encoder
    ↓
Classification
```

Topics such as large-scale ViT tuning, advanced attention variants, modern augmentation, transfer learning, pretraining, and architecture comparisons will be explored later in a separate computer-vision repository with more appropriate datasets.

### Final Architecture Comparison

The final comparison in this repository focuses on the **basic models**, not heavily optimized versions.

* [ ] Compare Softmax Regression, MLP, CNN, and ViT under a consistent evaluation setup
* [ ] Compare validation / test accuracy
* [ ] Compare parameter counts
* [ ] Compare training time where useful
* [ ] Summarize how the input representation changes across models
* [ ] Summarize strengths, limitations, and architectural differences
* [ ] Produce final project conclusions

The final goal is to make the progression clear:

```text
Softmax Regression
Raw pixels
    ↓
Linear mapping

        ↓ add nonlinear hidden representation

MLP
Flattened pixels
    ↓
Fully connected nonlinear representation

        ↓ introduce spatially structured operations

CNN
Image / feature maps
    ↓
Local convolutional processing
    ↓
Hierarchical spatial representation

        ↓ represent the image as tokens
          and use attention-based interaction

Vision Transformer
Image patches
    ↓
Patch embeddings
    ↓
Self-attention
    ↓
Contextual token representation
    ↓
Classification
```

Advanced CNN and Vision Transformer experimentation is intentionally deferred to a future repository with more complex datasets.
