# Convolutional Neural Network (CNN)

## Purpose

This document summarizes the core ideas needed to understand the CNN used in `notebooks/05_cnn.ipynb`.

The focus is on the architectural change from MLP to CNN:

```text
MLP
Image
  ↓
Flatten
  ↓
Fully Connected Network

CNN
Image
  ↓
Convolutional Feature Extractor
  ↓
Flatten
  ↓
Classifier
```

A CNN does not replace the entire classification pipeline. It adds a feature-extraction stage that preserves and uses the spatial structure of image data before classification.

---

## 1. Why CNN for Images?

A fully connected network usually flattens an image before processing it:

```text
28 × 28
   ↓
784
```

Flattening removes the explicit spatial relationship between neighboring pixels.

CNNs instead use:

```text
Local connections
Parameter sharing
Spatial feature extraction
```

This allows the network to learn local patterns such as edges, curves, and shapes while using fewer parameters than a fully connected layer over large images.

---

## 2. Convolution

A convolutional layer applies small learnable filters across local regions of the input.

```text
Input
  ↓
Sliding Kernel / Filter
  ↓
Feature Map
```

At each position, the filter performs element-wise multiplication with the local input region and sums the result.

Different filters can learn different features.

```text
Filter 1 → one local pattern
Filter 2 → another pattern
...
Filter N → N feature maps
```

In PyTorch:

```python
nn.Conv2d(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding
)
```

---

## 3. Stride and Padding

### Stride

Stride controls how far the kernel moves at each step.

```text
stride = 1
→ move one pixel at a time

stride = 2
→ move two pixels at a time
```

A larger stride reduces the spatial output size.

### Padding

Padding adds values, usually zeros, around the input.

It can:

- preserve spatial dimensions,
- reduce information loss near image borders.

For one spatial dimension:

```text
output =
floor((input + 2 × padding - kernel_size) / stride) + 1
```

Example used in this project:

```text
Input   = 28
Kernel  = 3
Padding = 1
Stride  = 1

Output  = 28
```

Therefore:

```text
1 × 28 × 28
      ↓
Conv2d(1 → 32, 3×3, padding=1)
      ↓
32 × 28 × 28
```

---

## 4. Channels and Feature Maps

CNN inputs are represented as:

```text
Batch × Channels × Height × Width
```

For MNIST:

```text
N × 1 × 28 × 28
```

because MNIST images are grayscale.

A convolutional filter spans all input channels.

The number of output channels is determined by the number of filters:

```text
32 filters
→ 32 output feature maps
```

For RGB images:

```text
Input channels = 3
```

so each convolutional filter also has depth 3.

---

## 5. ReLU

After convolution, ReLU introduces nonlinearity:

```text
ReLU(x) = max(0, x)
```

This allows stacked convolutional layers to learn nonlinear feature representations.

ReLU was already introduced in the MLP stage and is reused here.

---

## 6. Pooling

Pooling reduces the spatial size of feature maps.

This project uses Max Pooling:

```text
2 × 2 region
    ↓
keep maximum value
```

For example:

```text
32 × 28 × 28
      ↓
MaxPool2d(2)
      ↓
32 × 14 × 14
```

Pooling:

- reduces computation,
- compresses spatial information,
- makes the representation less sensitive to small shifts.

Unlike convolution, pooling has no learnable weights.

---

## 7. Flatten and Classifier

After convolution and pooling, the network contains a tensor of learned spatial features.

The tensor is flattened only near the end:

```text
64 × 7 × 7
    ↓
Flatten
    ↓
3136
```

The flattened features are then passed to a classifier:

```text
Learned Features
      ↓
Linear / MLP Classifier
      ↓
10 logits
```

This is the important distinction from the previous MLP:

```text
MLP
Flatten raw pixels first

CNN
Extract spatial features first
then flatten and classify
```

The classifier after the CNN feature extractor can be a single linear layer or a deeper MLP.

---

## 8. CNN Architecture Used in This Project

```text
Input
1 × 28 × 28
    ↓
Conv2d(1 → 32, 3×3, padding=1)
    ↓
ReLU
    ↓
MaxPool2d(2)
    ↓
32 × 14 × 14
    ↓
Conv2d(32 → 64, 3×3, padding=1)
    ↓
ReLU
    ↓
MaxPool2d(2)
    ↓
64 × 7 × 7
    ↓
Flatten
    ↓
3136
    ↓
Linear
    ↓
10 logits
```

The architecture can be divided into two parts:

```text
Feature Extractor
Conv → ReLU → Pool
Conv → ReLU → Pool

        ↓

Classifier
Flatten → Linear → 10 logits
```

---

## 9. What Changed from MLP?

### Reused

```text
MNIST
ReLU
Cross Entropy
SGD
Backpropagation
Train / Validation / Test
Evaluation
```

### Added

```text
Convolution
Kernels / Filters
Feature Maps
Channels
Stride
Padding
Pooling
Spatial Shape Tracking
```

The main architectural upgrade is therefore the representation stage:

```text
Softmax Regression
Raw pixels
    ↓
Linear classifier

MLP
Raw pixels
    ↓
Nonlinear fully connected representation
    ↓
Classifier

CNN
Image
    ↓
Convolutional spatial representation
    ↓
Classifier
```

---

## Key Takeaways

1. CNNs preserve image structure longer than fully connected networks.
2. Convolution learns local features using shared filters.
3. Multiple filters create multiple feature maps.
4. Stride and padding control spatial output size.
5. Pooling reduces spatial dimensions without learnable parameters.
6. Flatten connects the convolutional feature extractor to the classifier.
7. CNN is best viewed as an added spatial feature-extraction stage before classification.
8. The training pipeline remains largely the same as in the MLP stage.

---

## Reference

Primary reference used for this document:

**AI VIET NAM – AI Course 2025, _CNNs: Step-by-Step Examples_**  
Nguyễn Phúc Thịnh and Đinh Quang Vinh.

The reference covers the motivation for CNNs, convolution, stride, padding, pooling, flattening, image channels, and a step-by-step CNN forward-pass example.
