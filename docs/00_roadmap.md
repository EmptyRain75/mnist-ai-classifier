# Project Roadmap

**Status: the planned learning, implementation, experiment, and final-comparison scope is complete.**

This roadmap records the completed progression and its learning goals. See the [README](../README.md) for recorded results and [Final Comparison and Conclusion](07_final_comparison.md) for the final synthesis.

## 1. Roadmap Philosophy

This personal learning project follows a progressive, modular approach to image classification. Each stage builds on concepts from the previous stage, while supporting documents preserve the mathematics for future revision.

The central question is:

> What limitation or alternative design motivates the next model, and which components does it add, replace, or reuse?

The progression is Logistic Regression → Softmax Regression → PyTorch Softmax → MLP → CNN → Vision Transformer. It is a learning progression, not a promise of increasing accuracy at every stage.

Logistic Regression provides the binary-classification foundation. Softmax Regression on MNIST is the first main milestone. Structured experiments focus on Softmax and MLP; advanced CNN and ViT work belongs to a future project.

## 2. Completed Milestones

| Stage | Learning goal | Status |
|---|---|---|
| Foundations | Understand shapes, supervised learning, loss, and optimization | Complete |
| Logistic Regression | Build a complete binary classifier with NumPy | Complete |
| Softmax Regression | Extend linear classification to MNIST's ten classes | Complete |
| PyTorch Softmax | Understand what a framework automates | Complete |
| MLP | Introduce nonlinear hidden representations | Complete |
| CNN | Introduce local spatial processing and shared filters | Complete |
| Vision Transformer | Introduce patch tokens and attention | Complete |
| Softmax experiments | Study learning rate, batch size, and duration | Complete |
| MLP experiments | Study architecture, training choices, and gradient behavior | Complete |
| Final comparison | Summarize results, architectural changes, and limitations | Complete |

The completed order was basic architectures through ViT, followed by the structured Softmax and MLP experiments and final synthesis.

## 3. Modular Architecture

The classifier is divided conceptually into reusable responsibilities:

| Module | Responsibility | Examples of changes |
|---|---|---|
| Data | Load examples and define train/validation/test splits | Binary dataset → MNIST |
| Preprocessing | Prepare numeric inputs | Pixel scaling → training-derived standardization |
| Representation | Arrange the model input | Flat vector → image tensor → patch sequence |
| Feature processing | Build useful representations | Linear scoring → hidden layers → convolution → attention |
| Output | Produce class scores | One binary score → ten logits |
| Loss | Measure prediction error | Binary Cross Entropy → multiclass Cross Entropy |
| Optimization | Compute and apply parameter updates | Manual gradients → autograd; SGD or Adam |
| Evaluation | Measure performance and inspect behavior | Loss, accuracy, curves, predictions, and comparisons |

These are conceptual modules; each need not occupy a separate source file. The aim is to make reuse and architectural changes easy to explain.

## 4. Development Workflow

Each model stage followed a common cycle: study theory, build the learning notebook, inspect behavior, extract clean executable code, document the mathematics, and evaluate the results.

| Repository area | Role |
|---|---|
| `notebooks/` | Module-by-module explanations followed by code, outputs, visualizations, and observations |
| `src/` | Clean executable implementations extracted from the notebooks |
| `docs/` | Theory, mathematical derivations, architecture explanations, roadmap, and final synthesis |
| `experiments/` | Structured Softmax and MLP studies with recorded configurations and findings |

Exploratory notebook studies remain part of the learning record. The later experiment notebooks provide focused comparisons using shared training helpers.

## 5. Stage 0 — Foundations

**Status: complete.**

The reviewed concepts include:

- Vectors, matrices, shapes, transpose, dot products, and matrix multiplication.
- Features, labels, parameters, bias, predictions, and loss.
- Training, validation, and testing.
- Gradients, gradient descent, learning rate, epochs, and mini-batches.

These concepts support the implementations and explanations in the following stages.

## 6. Stage 1 — Logistic Regression

**Status: complete.**

A NumPy classifier was implemented on a small binary dataset. A linear score is passed through Sigmoid to obtain a probability, and Binary Cross Entropy provides the training objective.

The stage covered the forward pass, gradient derivation, parameter updates, evaluation, and decision-boundary visualization. It established the complete classification pipeline before extending it to multiple classes.

## 7. Stage 2 — Softmax Regression

**Basic implementation and structured experiments: complete.**

MNIST images are flattened from 28×28 pixels into 784-dimensional vectors. A linear layer produces ten logits. Softmax converts these scores to class probabilities; Cross Entropy measures prediction error.

The NumPy implementation made stable Softmax, gradients, and updates explicit. The PyTorch implementation introduced automatic differentiation and framework optimizers. In PyTorch, Cross Entropy receives raw logits directly, so the model does not need an explicit Softmax layer during training.

Completed studies in [01_softmax_experiments.ipynb](../experiments/01_softmax_experiments.ipynb):

- Learning rates: 0.01, 0.03, 0.1, and 0.3.
- Batch sizes: 32, 128, and 512.
- Training duration: a 20-epoch run inspected at multiple epochs.
- Validation-based configuration selection and final test evaluation.

The selected configuration used SGD at 0.1, batch size 32, and 14 epochs. It achieved **91.80% validation accuracy and 92.43% test accuracy** with 7,850 parameters.

Additional optimizer or regularization studies are optional extensions.

## 8. Stage 3 — MLP

**Basic implementation and structured experiments: complete.**

The baseline adds a nonlinear hidden representation: 784 inputs → 128 hidden units → ReLU → 10 logits. It reuses MNIST, Cross Entropy, mini-batch optimization, and evaluation.

Basic deliverables:

- `notebooks/04_mlp.ipynb`
- `src/04_mlp.py`
- `docs/04_mlp.md`

The studies in [02_mlp_experiments.ipynb](../experiments/02_mlp_experiments.ipynb) were guided by AI VIET NAM's **“Insight into Multi-layer Perceptron” by Quang-Vinh Dinh**.

| Topic | Completed comparison |
|---|---|
| Input normalization | Pixels in [0,1] versus training-derived standardization |
| Width | 64, 128, and 256 hidden units |
| Depth | One 128-unit layer versus two 128-unit layers |
| Activation | ReLU, Sigmoid, Tanh, and GELU |
| Initialization | Default, small normal, Xavier, and He-hidden schemes |
| BatchNorm | 256-unit ReLU model with and without BatchNorm |
| Optimizer | SGD at 0.1 versus Adam at 0.001 |
| Gradient behavior | Weight-gradient RMS in fresh shallow and six-hidden-layer networks |

The gradient probe used eight training batches without parameter updates. It showed strong attenuation in the deep Sigmoid network; it did not track gradients throughout training or demonstrate dying ReLU.

The selected model uses a 256-unit ReLU hidden layer, standardized inputs, BatchNorm, default initialization, SGD at 0.1, and batch size 32. Its saved epoch-8 checkpoint achieved **98.07% validation accuracy and 97.98% test accuracy** with 204,042 parameters.

The MLP still lacks explicitly local spatial processing and spatial weight sharing, motivating the CNN stage.

## 9. Stage 4 — CNN

**Basic architecture study: complete.**

The CNN preserves the image tensor and applies convolution, activation, and pooling before the classifier. Topics include channels, filters, stride, padding, feature maps, receptive fields, weight sharing, shape tracking, and hierarchical feature learning.

Completed deliverables:

- `notebooks/05_cnn.ipynb`
- `src/05_cnn.py`
- `docs/05_cnn.md`

The baseline has **50,186 parameters**, **98.72% validation accuracy**, and **98.95% test accuracy**—the highest recorded test accuracy in this project.

Deeper CNNs, extensive augmentation, transfer learning, and broader training studies remain outside the current scope.

## 10. Stage 5 — Vision Transformer

**Basic architecture study: complete.**

The ViT converts the image into sixteen 7×7 patches, projects them into 64-dimensional embeddings, prepends a CLS token, and adds learned positional embeddings. Two encoder blocks with four attention heads process the sequence; the final CLS representation produces ten logits.

Studied components include Q/K/V projections, multi-head self-attention, token-wise feed-forward networks, LayerNorm, pre-norm, residual connections, GELU, and Adam.

Completed deliverables:

- `notebooks/06_vit.ipynb`
- `src/06_vit.py`
- `docs/06_vit.md`

The baseline has **72,074 parameters**, **97.24% validation accuracy**, and **97.45% test accuracy**. It used Adam at 0.001 for ten epochs.

Large-scale tuning, attention variants, augmentation, pretraining, and transfer learning remain future work.

## 11. Architecture Evolution

| Transition | Main change |
|---|---|
| Logistic → Softmax | Extend binary classification to ten classes |
| NumPy Softmax → PyTorch Softmax | Automate gradients and updates while preserving the model family |
| Softmax → MLP | Add nonlinear hidden representations |
| MLP → CNN | Introduce local spatial processing and shared filters |
| CNN → ViT | Introduce patch tokens and attention-based interaction |

The CNN-to-ViT transition studies an alternative representation; it does not imply that ViT must outperform CNN on MNIST.

## 12. Reusable and Replaceable Components

| Component | Logistic | Softmax | MLP | CNN | ViT |
|---|---|---|---|---|---|
| Input representation | Feature vector | Flattened pixels | Flattened pixels | Image tensor | Patch-token sequence |
| Linear transformations | Binary score | Class scores | Hidden layers and head | Classification head | Patch projection, attention projections, FFN, and head |
| Main nonlinear processing | Sigmoid output | Softmax probabilities | ReLU baseline; other activations studied | ReLU | Attention Softmax and GELU |
| Spatial operations | — | — | — | Convolution and pooling | Patch extraction and positional embeddings |
| Internal normalization | — | — | BatchNorm in selected model | — in baseline | LayerNorm |
| Loss | Binary Cross Entropy | Multiclass Cross Entropy | Multiclass Cross Entropy | Multiclass Cross Entropy | Multiclass Cross Entropy |
| Optimization and evaluation concepts | Reused | Reused | Reused | Reused | Reused |

Flattening occurs at different points: Softmax and MLP flatten the input image, the CNN baseline flattens feature maps before its classifier, and ViT flattens individual patches for projection. All multiclass models can convert final logits to probabilities with Softmax when needed; PyTorch training uses logits directly.

## 13. Implementation Layers

**NumPy:** Logistic and Softmax Regression expose manual forward computation, gradients, and parameter updates.

**PyTorch:** Softmax, MLP, CNN, and ViT use modules, autograd, and optimizers. Mathematical explanations remain in `docs/`; separate NumPy implementations of the deeper architectures are not required.

## 14. Completed Experiment Method

Each structured study defined a question, held chosen settings fixed, ran fresh models with a seeded split and shuffle, recorded training/validation metrics, and documented the observed behavior.

The studies used a 50,000/10,000 MNIST training/validation split. Training-derived statistics were reused for validation and test preprocessing. Selected configurations and checkpoints were chosen using validation loss, then evaluated on the test set.

The studies were sequential and used one seed. Settings selected at one stage were carried into later studies; not every combination was explored. Changing batch size at fixed epochs also changes the number of updates.

Recorded cumulative runtimes were approximately **9.6 minutes for Softmax** and **18.9 minutes for MLP**, each below its separate two-hour budget. These include training and per-epoch validation; MLP also includes gradient diagnostics. Setup, downloads, plotting, and final test evaluation are excluded.

## 15. Final Model Comparison

**Status: complete** in [07_final_comparison.md](07_final_comparison.md).

The final document compares recorded validation/test metrics, parameter counts, representations, architectural assumptions, experiment findings, and limitations. Earlier architecture runs and later experiment-selected models are presented separately.

These runs do not use identical splits, preprocessing, optimizers, training durations, and selection methods across all models. They therefore support a comparison of the implemented configurations rather than a causal ranking of architectures. Runtime totals are experiment-budget records, not architecture speed benchmarks.

The final synthesis explains what each transition added and why more depth, parameters, or newer components did not automatically improve results.

## 16. Completion and Optional Future Work

- [x] Complete the foundations and binary-classification stage
- [x] Implement NumPy and PyTorch Softmax Regression
- [x] Complete the basic MLP, CNN, and ViT implementations
- [x] Write supporting architecture documentation and source scripts
- [x] Complete the planned Softmax experiments
- [x] Complete the planned MLP experiments and gradient diagnostic
- [x] Evaluate the selected experiment models on the test set
- [x] Write the final comparison and conclusions
- [x] Synchronize this roadmap with the completed learning scope

**No additional training is required for the current project scope.** Repository housekeeping—placing the final files in their intended paths, preserving executed notebook outputs, checking links and run instructions, and committing changes—does not add new learning milestones.

Optional future work includes repeated seeds, a dedicated dying-ReLU demonstration, broader regularization studies, matched architecture benchmarks, and CNN/ViT projects on more challenging datasets. These are extensions, not unfinished requirements of this roadmap.
