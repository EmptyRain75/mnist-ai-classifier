# Experiments

This folder is reserved for deeper and more systematic experiments performed after the basic model progression has been completed.

The main priority is to first build and understand the basic architectures:

```text
Softmax Regression
        ↓
MLP
        ↓
CNN
        ↓
Vision Transformer
```

The goal of this folder is to revisit those models later and investigate **why modern training techniques are needed, what problems they solve, and how they affect optimization and generalization**.

---

## Planned Experiments

# 1. Softmax Regression — Hyperparameter Study

Systematically investigate how training configuration affects Softmax Regression.

Possible variables include:

```text
Learning rate
Batch size
Number of epochs
Optimizer
Regularization
```

Observe:

```text
Training loss
Validation loss
Training accuracy
Validation accuracy
Convergence speed
Training time
```

The purpose is to understand how hyperparameters affect optimization before moving to deeper models.

---

# 2. MLP — Vanishing Gradient Study

Construct a sufficiently deep MLP to deliberately demonstrate the **vanishing gradient problem**.

Rather than relying only on final accuracy, directly measure gradient flow through the network.

```text
Layer 1 → gradient norm
Layer 2 → gradient norm
Layer 3 → gradient norm
...
Layer N → gradient norm
```

Then investigate different techniques that improve deep-network training.

---

## 2.1 Sigmoid vs ReLU

Compare deep networks using:

```text
Sigmoid
   vs
ReLU
```

Sigmoid can saturate for large positive or negative inputs, causing derivatives to become very small.

Repeated multiplication by small derivatives during backpropagation can cause gradients in early layers to shrink.

ReLU avoids saturation on the positive side:

```text
ReLU(x) = max(0, x)
```

Observe:

```text
Gradient norms
Activation distributions
Training loss
Validation loss
Convergence speed
Accuracy
```

---

## 2.2 Xavier vs He / Kaiming Initialization

Compare different initialization strategies:

```text
Default / poor initialization
        vs
Xavier initialization
        vs
He / Kaiming initialization
```

### Xavier Initialization

Designed to keep activation and gradient scales relatively stable across layers.

Commonly used with activations such as:

```text
Sigmoid
Tanh
```

### He / Kaiming Initialization

Designed for ReLU-style activations.

Commonly used with:

```text
ReLU
Leaky ReLU
```

Observe whether better initialization improves:

```text
Gradient flow
Activation scale
Training stability
Convergence
Final performance
```

---

## 2.3 Batch Normalization

Compare:

```text
Without BatchNorm
        vs
With BatchNorm
```

Batch Normalization normalizes intermediate activations during training and introduces learned scale and shift parameters.

Investigate its effect on:

```text
Activation distributions
Gradient flow
Training stability
Convergence speed
Learning-rate sensitivity
Validation performance
```

The goal is to understand why BatchNorm often makes deeper networks easier to train.

---

## 2.4 SGD vs Adam

Compare:

```text
SGD
 vs
Adam
```

Observe:

```text
Training loss
Validation loss
Convergence speed
Gradient behavior
Final performance
```

Adam may improve optimization when gradients are small, noisy, or differently scaled across parameters.

However, Adam should not automatically be treated as a direct solution to the underlying vanishing-gradient problem.

---

## 2.5 Gradient Norm Monitoring

Track gradient magnitude throughout the network.

For each layer:

```text
Layer 1 → ||∇W₁||
Layer 2 → ||∇W₂||
Layer 3 → ||∇W₃||
...
Layer N → ||∇Wₙ||
```

This provides a direct way to observe gradient behavior.

```text
Vanishing gradients
→ gradient norms approach 0

Exploding gradients
→ gradient norms become very large
```

Possible visualizations:

```text
Gradient norm vs layer depth
Gradient norm vs epoch
Gradient norm distribution
```

This will be one of the main tools used to **demonstrate** the vanishing-gradient problem instead of only inferring it from poor accuracy.

---

## 2.6 Gradient Clipping / Gradient Normalization

After monitoring gradient norms, investigate techniques for controlling unstable gradients.

Possible methods:

```text
Gradient clipping by value
Gradient clipping by norm
Gradient normalization
```

Conceptually:

```text
if ||g|| > threshold:
    scale gradient down
```

This is primarily useful for controlling **exploding gradients**.

The experiment should distinguish between:

```text
Gradient monitoring
→ measurement

Gradient clipping / normalization
→ intervention
```

---

## 2.7 Residual / Skip Connections

Compare a normal deep network with a network containing residual connections.

### Standard Network

```text
x
↓
Layer
↓
Layer
↓
Layer
↓
Output
```

### Residual Network

```text
x ───────────────┐
↓                │
Layer            │
↓                │
Layer            │
↓                │
+  ←─────────────┘
↓
Output
```

Instead of learning only:

```text
H(x)
```

a residual block learns:

```text
H(x) = F(x) + x
```

Residual connections provide shorter paths for both information and gradients.

This introduces the core idea behind **ResNet**.

Observe:

```text
Gradient flow
Training stability
Convergence
Depth scalability
Validation performance
```

---

## 2.8 Deep Supervision / Auxiliary Losses

Normally, the final loss must propagate backward through the entire network:

```text
Input
 ↓
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
Output
 ↓
Main Loss
```

With **deep supervision**, intermediate layers receive additional prediction heads and auxiliary losses.

```text
Layer 1
   ↓
Layer 2 ─────→ Auxiliary Head → Auxiliary Loss
   ↓
Layer 3
   ↓
Layer 4 ─────→ Auxiliary Head → Auxiliary Loss
   ↓
Output
   ↓
Main Loss
```

The total objective can combine the losses:

```text
Total Loss
=
Main Loss
+
λ₁ × Auxiliary Loss 1
+
λ₂ × Auxiliary Loss 2
```

This gives earlier layers a more direct learning signal instead of requiring all gradient information to travel only from the final output.

This technique may also be referred to as:

```text
Deep Supervision
Auxiliary Losses
Auxiliary Classifiers
```

Compare it with residual connections, since both can make deep networks easier to train, but they do so in different ways.

---

## 2.9 Layer-wise Training

Investigate training layers or blocks progressively instead of optimizing the entire deep network at once.

Possible workflow:

```text
Train early layers
        ↓
Freeze learned layers
        ↓
Add / train deeper layers
        ↓
Unfreeze
        ↓
Joint fine-tuning
```

This is more accurately described as:

```text
Layer-wise training
Layer-wise pretraining
Progressive freezing / unfreezing
```

rather than standard fine-tuning.

The experiment can investigate whether providing simpler learning stages improves optimization in deep networks.

---

# Experiment Philosophy

The goal is not simply to find which configuration gives the highest accuracy.

Each experiment should answer a specific question:

```text
What problem occurs?
        ↓
Why does it occur?
        ↓
How can it be measured?
        ↓
What technique may address it?
        ↓
What changes after applying the technique?
```

Where possible, experiments should change **one main variable at a time** while keeping the rest of the setup fixed.

Useful measurements include:

```text
Training loss
Validation loss
Training accuracy
Validation accuracy
Test accuracy
Gradient norms
Activation distributions
Convergence speed
Parameter count
Training time
Inference time
```

---

# Current Priority

These experiments are intentionally postponed until the basic architecture progression is complete.

```text
CURRENT

Basic MLP ✓
    ↓
Basic CNN
    ↓
Basic Vision Transformer
    ↓
Understand the complete model progression
```

Then return to this folder for deeper experiments:

```text
LATER

Structured Experiments
    ↓
Softmax Hyperparameters
    ↓
Vanishing Gradient Demonstration
    ↓
Sigmoid vs ReLU
    ↓
Xavier / He Initialization
    ↓
Batch Normalization
    ↓
SGD vs Adam
    ↓
Gradient Norm Monitoring
    ↓
Gradient Clipping / Normalization
    ↓
Residual Connections
    ↓
Deep Supervision / Auxiliary Losses
    ↓
Layer-wise Training
```

The purpose of this folder is therefore to move from:

```text
"How do I build the model?"
```

to:

```text
"Why is the model difficult to train,
and what techniques make deep learning work?"
```
