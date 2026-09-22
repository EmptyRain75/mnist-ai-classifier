# Multilayer Perceptron

## 1. Motivation

Softmax Regression maps the flattened MNIST input directly to class logits:

```text
784
 ↓
Linear
 ↓
10 logits
```

This is still a linear classifier with respect to the original input features.

The MLP introduces a hidden nonlinear representation:

```text
784
 ↓
Linear
 ↓
Hidden Layer
 ↓
ReLU
 ↓
Linear
 ↓
10 logits
```

The purpose of the hidden layer is to learn an intermediate representation of the input before producing the final class scores.

---

## 2. Forward Pass

For a one-hidden-layer MLP:

$$
Z_1 = XW_1 + b_1
$$

$$
H = \text{ReLU}(Z_1)
$$

$$
Z_2 = HW_2 + b_2
$$

where:

- $X$ is the input.
- $Z_1$ is the hidden-layer pre-activation.
- $H$ is the hidden representation.
- $Z_2$ contains the output logits.

For MNIST with hidden size $H=128$:

```text
X   : N × 784
W1  : 784 × 128
b1  : 128
H   : N × 128
W2  : 128 × 10
b2  : 10
Z2  : N × 10
```

---

## 3. Why Nonlinearity Is Necessary

Stacking two linear transformations without an activation function does not create a more expressive model.

Suppose:

$$
H = XW_1+b_1
$$

$$
Z = HW_2+b_2
$$

Then:

$$
Z = (XW_1+b_1)W_2+b_2
$$

which can be rewritten as:

$$
Z = XW' + b'
$$

Therefore, multiple linear layers without a nonlinear activation are still equivalent to a single linear transformation.

The ReLU activation:

$$
\text{ReLU}(z)=\max(0,z)
$$

prevents this collapse and allows the network to represent nonlinear functions.

---

## 4. Width and Depth

Two important architectural properties are:

```text
Width
= number of neurons in a hidden layer

Depth
= number of hidden layers
```

Increasing width generally increases model capacity by introducing more hidden units.

Increasing depth allows the network to compose multiple nonlinear transformations.

Neither greater width nor greater depth guarantees better validation performance.

---

## 5. Model Capacity

Compared with Softmax Regression, an MLP contains more trainable parameters and can represent more complex functions.

For example:

```text
Softmax Regression
784 → 10

MLP
784 → 128 → 10
```

The MLP therefore has greater representational capacity, but also:

- higher computational cost
- more hyperparameters
- greater potential to fit the training data more strongly than the validation data

---

## 6. Backpropagation

The forward pass is:

```text
X
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Loss
```

During backpropagation, gradients flow in the reverse direction:

```text
Loss
 ↓
Output Linear
 ↓
ReLU
 ↓
Hidden Linear
```

PyTorch automatically computes these gradients using autograd.

The training pattern remains:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 7. Transition from Softmax Regression

### Reused

```text
MNIST
Flattening
Mini-batches
Cross Entropy
SGD
Train / validation / test split
Accuracy evaluation
```

### Added

```text
Hidden layer
ReLU activation
Nonlinear representation
```

The main transition is:

```text
Softmax Regression
Linear classifier
        ↓
Add hidden representation
Add nonlinearity
        ↓
MLP
Nonlinear classifier
```

---

## 8. Limitation of the MLP

Although the MLP improves classification performance, MNIST images are still flattened:

```text
28 × 28
 ↓
784
```

The network therefore does not explicitly preserve the two-dimensional spatial structure of the image.

This limitation motivates the next stage:

```text
MLP
 ↓
CNN
```

where convolution will be used to learn spatial image features directly.
