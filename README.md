# MNIST Image Classification — From Scratch to Deep Learning

This is a personal learning project: a progressive study of image classification, starting from simple linear models and moving toward modern deep learning architectures.

Its purpose is to demonstrate my understanding of Softmax Regression, MLP, CNN, and Vision Transformer through their implementations, architectural ideas, and underlying mathematics. The supporting documents in `docs/` record the theory and mathematical details I have studied and serve as reference material for future revision.

**The basic architecture progression through ViT and the planned Softmax and MLP experiments are complete.** The remaining work is the final written synthesis and repository review.

The main goal is not only to achieve higher accuracy, but to understand:

- how each model works,
- what limitation motivates the next model,
- which parts of the pipeline are reused,
- how the data representation changes,
- and how architecture and training choices affect learning.

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

Softmax Regression, MLP, CNN, and Vision Transformer are developed primarily on **MNIST**. Logistic Regression introduces the binary-classification foundations on a separate binary dataset.

## Experiment Scope

The basic architecture progression continues through CNN and Vision Transformer, while the systematic experiments in this repository stop at the **MLP stage**.

| Stage | Scope |
|---|---|
| Softmax Regression | Learning rate, batch size, and training duration |
| MLP | Input normalization, width/depth, activations, initialization, BatchNorm, optimizers, and gradient diagnostics |
| CNN | Basic architecture implementation and evaluation |
| Vision Transformer | Basic architecture implementation and evaluation |

MLP is the main experimental model. Its experiments are guided by the AI VIET NAM course material **“Insight into Multi-layer Perceptron” by Quang-Vinh Dinh**.

Advanced CNN and ViT experiments are intentionally deferred to a future computer-vision repository using more complex datasets. Additional Softmax optimizer/regularization studies and a dedicated dying-ReLU demonstration are optional extensions, rather than unfinished requirements of the completed experiment stage.

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
- `notebooks/` — step-by-step learning, implementation, visualizations, and observations.
- `src/` — clean executable implementations extracted from the notebooks.
- `experiments/` — systematic Softmax and MLP studies, including configurations, outputs, and conclusions.

See [the project roadmap](docs/00_roadmap.md) for the architecture progression, milestones, and development workflow.

The completed experiment notebooks are:

- [Softmax experiments](experiments/01_softmax_experiments.ipynb)
- [MLP experiments](experiments/02_mlp_experiments.ipynb)

## Recorded Results

### Basic Architecture Runs

These are the previously recorded architecture runs, kept separate from the later experiment-selected configurations.

| Model | Parameters | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Softmax Regression — NumPy, earlier selected run | 7,850 | 92.05% | 92.54% |
| MLP — 128 hidden units, earlier baseline | 101,770 | 97.11% | 97.44% |
| CNN — baseline | 50,186 | 98.72% | 98.95% |
| ViT — baseline | 72,074 | 97.24% | 97.45% |

### Configurations Selected in the Experiment Notebooks

| Model | Selected configuration | Parameters | Validation loss | Validation accuracy | Test loss | Test accuracy |
|---|---|---:|---:|---:|---:|---:|
| Softmax — PyTorch | SGD 0.1, batch 32, 14 epochs | 7,850 | 0.2951 | 91.80% | 0.2704 | 92.43% |
| MLP — PyTorch | 256 hidden units, ReLU, standardized inputs, BatchNorm, default initialization, SGD 0.1, batch 32, epoch-8 checkpoint | 204,042 | 0.0679 | 98.07% | 0.0638 | 97.98% |

Configurations and checkpoints in the experiment notebooks were selected using **validation loss**. Their selected models were then evaluated on the test set. The selected MLP checkpoint includes BatchNorm running statistics, and evaluation uses the same training-derived input-standardization statistics.

The new MLP experiment baseline achieved 97.18% final validation accuracy; the earlier architecture baseline above achieved 97.11%. They are separate recorded runs.

### How to Interpret These Results

- This is a summary of learning runs, not a controlled experiment changing architecture alone.
- The earlier NumPy Softmax run used a different split procedure. PyTorch studies use a seeded 50,000/10,000 training/validation split.
- Optimizers, preprocessing, training durations, initialization, and checkpoint-selection procedures differ across runs. ViT used Adam; the earlier MLP and CNN baselines used SGD.
- Experiment studies changed one chosen setting at a time, then carried selected settings forward. They used one seed and did not exhaustively search all combinations or interactions.
- CNN achieved the highest recorded test accuracy. Its useful spatial inductive biases are a plausible contributor on MNIST, but these runs do not isolate dataset size or prove a general CNN-versus-ViT ranking.
- More parameters, more depth, or a newer architecture did not automatically produce better results.

The cumulative runtime ledgers recorded approximately **9.6 minutes for Softmax** and **18.9 minutes for MLP**, each below its two-hour budget. These include training and per-epoch validation; the MLP ledger also includes gradient diagnostics. Setup, downloads, plotting, and final test evaluation are excluded. These totals are not per-model architecture timing benchmarks.

## Current Progress

### Project Setup

- [x] Create GitHub repository
- [x] Create project folder structure
- [x] Create initial README
- [x] Create roadmap
- [x] Set up first development notebook

### Foundations

- [x] Review required linear algebra
- [x] Review supervised learning
- [x] Understand parameters and bias
- [x] Understand loss functions
- [x] Understand gradient descent
- [x] Understand train / validation / test

### Logistic Regression

- [x] Theory
- [x] Sigmoid
- [x] Binary Cross Entropy
- [x] Gradient derivation
- [x] NumPy implementation
- [x] Train on binary dataset
- [x] Visualize decision boundary
- [x] Evaluate model
- [x] Document results

### Softmax Regression — Basic Implementation

- [x] Multiclass classification
- [x] Softmax
- [x] Numerical stability
- [x] Cross Entropy
- [x] Gradient derivation
- [x] NumPy implementation
- [x] Train on MNIST
- [x] Evaluate model
- [x] PyTorch implementation
- [x] NumPy vs PyTorch comparison

### Softmax Regression — Simple Experiments

- [x] Establish a fixed baseline configuration
- [x] Experiment with learning rate
- [x] Experiment with batch size
- [x] Experiment with training duration
- [x] Compare training and validation behavior
- [x] Select a configuration using validation loss
- [x] Evaluate the selected configuration on the test set
- [x] Document observations and conclusions

**Findings:** learning rate 0.1 gave the best five-epoch validation results among the tested rates. Batch size 32 outperformed larger batches under the fixed learning rate and epoch budget, while performing more updates. Longer training produced modest improvements; the selected duration was 14 epochs.

### MLP — Basic Implementation

- [x] Build a basic MLP for MNIST
- [x] Add hidden layer(s) and nonlinear activation
- [x] Train and evaluate the basic model
- [x] Compare hidden-layer widths
- [x] Compare different network depths
- [x] Compare the MLP against Softmax Regression

### MLP — Experiments

The experiment plan was guided by:

```text
AI VIET NAM – AI Course 2025
"Insight into Multi-layer Perceptron"
Quang-Vinh Dinh
```

- [x] Compare pixels in [0,1] with training-derived input standardization
- [x] Study hidden-layer widths: 64, 128, and 256
- [x] Study depth: one 128-unit layer versus two 128-unit layers
- [x] Compare ReLU, Sigmoid, Tanh, and GELU
- [x] Compare default, small-normal, Xavier, and He-hidden initialization
- [x] Experiment with Batch Normalization
- [x] Compare SGD at 0.1 with Adam at 0.001
- [x] Inspect weight-gradient RMS in fresh shallow and deep MLPs
- [x] Discuss how activation, initialization, and depth affect gradient behavior
- [x] Save and select checkpoints using validation loss
- [x] Evaluate the selected checkpoint on the test set
- [x] Document experiments and conclusions

**Findings:** standardized inputs and a wider 256-unit hidden layer improved validation performance in the tested runs. ReLU and GELU performed similarly, while default initialization remained competitive. BatchNorm improved the best validation checkpoint, and SGD at 0.1 achieved lower best validation loss than Adam at 0.001.

The gradient diagnostic showed strong attenuation toward earlier layers in a fresh six-layer Sigmoid network. It used eight training batches with no parameter updates; it did not monitor gradients throughout training or demonstrate dying ReLU in the trained models.

### CNN — Basic Implementation

- [x] Study convolution, kernels, feature maps, stride, padding, and pooling
- [x] Build a basic CNN in PyTorch
- [x] Keep the original 2D image representation
- [x] Train the CNN on MNIST
- [x] Evaluate the basic model
- [x] Compare CNN against MLP and Softmax Regression
- [x] Document the basic CNN architecture
- [x] Refactor basic CNN components into `src/`

### CNN — Advanced Experiments

Advanced CNN experimentation is **outside the scope of this MNIST repository**.

The CNN stage studies local connectivity, weight sharing, convolution, pooling, spatial feature maps, and hierarchical feature extraction. Deeper architectures, ResNet-style models, extensive augmentation, transfer learning, and modern CNN training strategies are deferred to a future computer-vision repository.

### Vision Transformer — Basic Implementation

- [x] Study image patches and patch embeddings
- [x] Study the classification token and positional embeddings
- [x] Study self-attention and multi-head attention
- [x] Study Transformer encoder blocks
- [x] Study LayerNorm, pre-norm, residual connections, and GELU
- [x] Study Adam as the ViT training optimizer
- [x] Build a basic ViT for MNIST in PyTorch
- [x] Train and evaluate the basic ViT
- [x] Compare ViT against CNN, MLP, and Softmax Regression
- [x] Document the basic ViT architecture
- [x] Refactor basic ViT components into `src/`

The baseline uses sixteen 7×7 image patches, embedding dimension 64, one CLS token, four attention heads, and two encoder blocks. Its final CLS representation produces ten class logits.

### Vision Transformer — Advanced Experiments

Advanced ViT experimentation is **outside the scope of this MNIST repository**.

Large-scale tuning, advanced attention variants, modern augmentation, transfer learning, pretraining, and further architecture comparisons are deferred to a future computer-vision repository with more appropriate datasets.

### Final Architecture Comparison and Wrap-Up

The recorded-results tables above summarize the basic runs and the experiment-selected models separately. A final synthesis will explain the architectural progression and learning outcomes, with the comparison limits stated explicitly.

- [x] Summarize recorded validation/test accuracy
- [x] Summarize parameter counts
- [x] Separate architecture baselines from experiment-selected configurations
- [x] State differences in training settings and checkpoint selection
- [x] Record experiment runtime boundaries
- [x] Write the final comparison and project conclusions in `docs/`
- [x] Synchronize `docs/00_roadmap.md` with completed milestones
- [ ] Verify executed experiment notebooks, saved outputs, and final conclusions are committed
- [ ] Verify file links, dependencies, and run instructions in the final repository

No additional training is required for this planned scope. Repeated seeds, broader hyperparameter searches, or strictly matched architecture benchmarks would be optional extensions.

## What Each Architectural Transition Added

| Transition | Main addition |
|---|---|
| Logistic → Softmax | Extend binary classification to multiple classes |
| Softmax → MLP | Learn nonlinear hidden representations from flattened pixels |
| MLP → CNN | Introduce local spatial processing and shared filters |
| CNN → ViT | Represent images as patch tokens and exchange information through attention |

Each MNIST model produces ten logits for Cross Entropy classification, while its representation and feature-processing modules change. The models illustrate different ways to learn from images; the progression does not guarantee increasing accuracy at every stage.
