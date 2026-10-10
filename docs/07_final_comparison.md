# Final Comparison and Conclusion

This project studied image classification progressively, from Logistic Regression to Softmax Regression, MLP, CNN, and Vision Transformer. The basic implementations were followed by focused Softmax and MLP experiments.

The strongest recorded MNIST result was the baseline CNN at **98.95% test accuracy**. The experiment-selected MLP achieved **97.98%**, while the basic ViT achieved **97.45%**. These results describe the implemented configurations; the main learning outcome is understanding how representations, architecture, and training choices affect a model.

## 1. What Changed Across Architectures?

| Model | Input representation | Learned processing | Main contribution | Main limitation in this project |
|---|---|---|---|---|
| Logistic Regression | Feature vector | Linear score followed by Sigmoid | Establishes binary classification, loss, and gradients | Linear decision boundary; binary task |
| Softmax Regression | 784 flattened pixels | Ten linear class scores | Extends classification to ten MNIST classes | Cannot learn nonlinear hidden features |
| MLP | 784 flattened pixels | Fully connected layers and nonlinear activations | Learns nonlinear combinations of pixels | Does not explicitly build in local spatial processing or spatial weight sharing |
| CNN | Two-dimensional image | Convolution, activation, and pooling | Learns reusable local features while preserving spatial structure | This small baseline does not explore deeper architectures or harder datasets |
| ViT | Sequence of image-patch embeddings | Self-attention and token-wise feed-forward networks | Learns interactions across patch tokens | This small model trained from scratch did not match the CNN result |

**Logistic → Softmax:** the task changes from binary classification to multiple classes. The model remains linear in its input features.

**Softmax → MLP:** nonlinear hidden layers let the model learn features instead of scoring only the raw pixels. Adding linear layers without nonlinear activations would still produce a linear mapping.

**MLP → CNN:** local filters and weight sharing introduce assumptions that suit images. A filter can detect a similar stroke at different positions without learning separate weights for every location. Flattening preserves pixel values and their fixed ordering, but a dense layer does not explicitly exploit neighboring pixels in this way.

**CNN → ViT:** the image becomes a sequence of patches. Our ViT uses sixteen 7×7 patches, 64-dimensional embeddings, a CLS token, learned positional embeddings, and two four-head encoder blocks. Attention combines information across tokens; the final CLS representation feeds the classifier. LayerNorm, residual connections, and GELU support this processing.

The transition to PyTorch changes implementation mechanics—automatic differentiation, modules, and device execution—rather than changing Softmax Regression into a different model family.

## 2. What Stayed the Same?

The MNIST models share the same overall learning pipeline:

1. Prepare training, validation, and test data.
2. Produce ten class logits from each image.
3. Compute Cross Entropy against the correct class.
4. Calculate gradients and update parameters using mini-batches.
5. Inspect validation behavior and evaluate the selected model on test data.

The representation and feature-processing modules change most substantially. The classification objective and evaluation concepts remain reusable. In the PyTorch implementations, Cross Entropy receives raw logits directly.

## 3. Recorded Model Results

### Earlier Architecture Runs

| Model | Parameters | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| Softmax — NumPy, earlier selected run | 7,850 | 92.05% | 92.54% |
| MLP — 128 hidden units | 101,770 | 97.11% | 97.44% |
| CNN — baseline | 50,186 | 98.72% | 98.95% |
| ViT — baseline | 72,074 | 97.24% | 97.45% |

Logistic Regression used a separate binary dataset, so its accuracy is not included in the MNIST comparison.

### Models Selected in the Experiment Notebooks

| Metric | Softmax — PyTorch | MLP — PyTorch |
|---|---:|---:|
| Parameters | 7,850 | 204,042 |
| Validation loss | 0.2951 | 0.0679 |
| Validation accuracy | 91.80% | 98.07% |
| Test loss | 0.2704 | 0.0638 |
| Test accuracy | 92.43% | 97.98% |

**Selected Softmax:** SGD at 0.1, batch size 32, pixels in [0,1], and 14 epochs. Epoch 14 had the lowest validation loss in the duration study. The configuration was retrained because that study had retained only its final model.

**Selected MLP:** one 256-unit ReLU hidden layer, training-derived input standardization, BatchNorm before ReLU, default initialization, SGD at 0.1, and batch size 32. The saved epoch-8 checkpoint had the lowest validation loss. Evaluation restored both learned parameters and BatchNorm running statistics and reused the training-derived preprocessing values.

The new MLP experiment baseline achieved 97.18% final validation accuracy. It is a separate run from the earlier 97.11% baseline listed above; those values should not be merged into one run.

### Reading the Comparison

The CNN achieved the highest recorded accuracy with fewer parameters than either MLP configuration or the ViT. This demonstrates that parameter count alone does not determine performance. Its local processing and shared filters are useful architectural assumptions for recognizing digit strokes.

The CNN exceeded the ViT by **1.50 percentage points** in test accuracy. Limited training data is a plausible contributor, but the experiment did not vary dataset size. Training duration, optimizer, patch configuration, and other design choices could also contribute. This result does not establish that CNNs generally outperform ViTs.

The selected MLP exceeded the earlier basic MLP by **0.54 percentage points** in recorded test accuracy. Multiple choices changed, including width, preprocessing, BatchNorm, and checkpoint selection, so the improvement cannot be assigned to any one of them.

The earlier NumPy Softmax result was slightly higher than the selected PyTorch result. Since the split procedure and training configuration differ, this is not evidence that NumPy produces better models than PyTorch.

## 4. What the Softmax Experiments Showed

| Study | Observed result | Interpretation |
|---|---|---|
| Learning rate: 0.01, 0.03, 0.1, 0.3 | 0.1 gave the best final five-epoch validation results | Smaller rates learned more slowly; 0.3 did not improve the final result |
| Batch size: 32, 128, 512 | Batch 32 reached 91.50% validation accuracy, versus 90.71% and 89.15% | Equal epochs do not mean equal optimizer updates; batch 32 performed 7,815 updates, versus 490 for batch 512 |
| Duration: up to 20 epochs | Accuracy improved from 91.50% at epoch 5 to 91.89% at epoch 20; minimum validation loss occurred at epoch 14 | Extra training gave diminishing gains; loss and accuracy need not select the same epoch |

These studies establish a simple lesson: learning rate, batch size, and duration interact. A larger batch may need a different learning rate or more epochs. The selected settings are the result of a small sequential study, not a global optimum.

## 5. What the MLP Experiments Showed

| Study | Main observation | Takeaway |
|---|---|---|
| Input standardization | Final validation accuracy increased from 97.18% to 97.66% for the 128-unit model | Input scaling affected learning at the fixed learning rate |
| Width | The 256-unit model reached 97.91%, versus 97.66% for 128 and 97.21% for 64 | Extra width helped in these runs |
| Depth | Two 128-unit layers reached 97.42%, below one 128-unit layer | Extra depth did not automatically help with the shared settings |
| Activation | ReLU had best validation loss 0.0717; GELU had 0.0730, with slightly higher final accuracy | Similar accuracy can accompany different Cross Entropy; retain a consistent selection criterion |
| Initialization | Default initialization had the lowest best validation loss among the tested schemes | Lower first-epoch training loss with Xavier or He did not guarantee better generalization |
| BatchNorm | Best validation loss improved from 0.0717 to 0.0679; the best checkpoint reached 98.07% accuracy | Evaluate saved checkpoints rather than assuming the last epoch is best |
| Optimizer | SGD at 0.1 achieved best loss 0.0679; Adam at 0.001 achieved 0.0740 | These optimizer configurations differ; Adam is not an automatic improvement |

Input standardization used statistics from training pixels only. BatchNorm acts on hidden features: it uses mini-batch statistics during training and running statistics during evaluation. These are distinct operations.

For the selected MLP, epoch 8 was better by validation loss than epoch 10. This illustrates why continuing to reduce training loss does not ensure improved validation performance. Accuracy measures whether the winning class is correct; Cross Entropy also reflects the probability assigned to the correct class, so their best epochs can differ.

### Gradient Diagnostic

Fresh networks were inspected using the same eight training batches, width 128, default initialization, and no optimizer updates. Deep networks had six hidden layers. Mean weight-gradient RMS was measured separately for each layer.

| Fresh network | First hidden layer RMS | Last hidden layer RMS |
|---|---:|---:|
| ReLU, six hidden layers | 7.765e-5 | 2.333e-4 |
| Sigmoid, six hidden layers | 1.471e-7 | 1.196e-3 |
| Tanh, six hidden layers | 5.021e-4 | 7.805e-4 |

The deep Sigmoid network's first-layer gradient RMS was about **8,100 times smaller** than its last hidden layer's. Tanh's hidden-layer magnitudes were more uniform in this probe. The one-hidden-layer ReLU network had first-layer RMS 7.251e-3, much larger than the deep ReLU network's first-layer value.

These observations show that depth and activation can strongly affect gradient magnitudes at initialization. They do not track gradients throughout training, demonstrate dying ReLU, or explain the complete training-performance ranking. Weight-gradient magnitudes also depend on layer inputs and the loss.

## 6. Runtime and Comparison Limits

| Experiment notebook | Recorded cumulative runtime | Included |
|---|---:|---|
| Softmax | About 9.6 minutes | Training and per-epoch validation |
| MLP | About 18.9 minutes | Training, per-epoch validation, and gradient diagnostics |

Both stayed below their separate two-hour budgets on a Tesla T4 runtime. These are cumulative ledger totals, including recorded repeat runs, rather than the cost of training one model. Setup, downloading, plotting, and final test evaluation are excluded. Earlier model timings used different boundaries or execution conditions, so an architecture speed ranking would not be justified.

The main limits of the conclusions are:

- **Single-seed runs:** small differences may change with initialization or data order; no uncertainty intervals were estimated.
- **Sequential choices:** selected settings were carried forward. The studies do not cover every interaction among width, activation, initialization, and optimizer.
- **Different architecture configurations:** the NumPy split procedure, preprocessing, optimizers, training duration, and selection methods are not identical across all models.
- **Repeated validation use:** validation guided many decisions, so its best observed result may be optimistic. Test results were reported after selecting each experiment's final configuration and were not used to choose its settings.
- **One small image task:** MNIST results do not establish performance on larger color images or more diverse datasets.

## 7. Final Conclusion

The project completed a modular progression from linear classification to nonlinear, convolutional, and attention-based representations. Building the early models from scratch made the forward pass, loss, gradients, and parameter updates explicit. The PyTorch stages reused those concepts while allowing more complex architectures.

The experiments showed that preprocessing, capacity, initialization, normalization, optimization, and model selection all matter. They also showed why greater depth, more parameters, and newer components should be evaluated rather than assumed to improve a model.

The CNN achieved the strongest recorded MNIST result, while the MLP experiments provided the most detailed study of training choices. The ViT completed the architectural learning progression by introducing patch tokens and attention-based interaction.

The planned implementation and experiment scope is complete. Future CNN and ViT work can use more challenging datasets and investigate augmentation, transfer learning, deeper architectures, and more controlled repeated-run comparisons. No additional training is required to complete this repository's current learning goals.

## Related Project Material

- [Project overview and progress](../README.md)
- [Project roadmap](00_roadmap.md)
- [Softmax experiment notebook](../experiments/01_softmax_experiments.ipynb)
- [MLP experiment notebook](../experiments/02_mlp_experiments.ipynb)

The MLP study topics were guided by AI VIET NAM's *Insight into Multi-layer Perceptron* by Quang-Vinh Dinh. Numerical results in this document come from this project's recorded runs.
