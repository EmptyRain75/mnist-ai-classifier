# Convolutional Neural Network (CNN)

## Purpose

This document is the **theory and mathematics companion** to `notebooks/05_cnn.ipynb`.

The notebook stays intentionally simple:

```text
Load MNIST
    ↓
Define CNN
    ↓
Train with PyTorch
    ↓
Evaluate
    ↓
Visualize predictions / feature maps
```

The notebook answers:

> **How do I build and train this CNN?**

This document answers:

> **What is each layer calculating, and what does `loss.backward()` do mathematically?**

It covers:

- convolution,
- stride and padding,
- channels and feature maps,
- parameter counting,
- ReLU,
- pooling,
- flattening,
- the linear classifier,
- Softmax + Cross Entropy,
- and the backward pass through every important operation.

### Reference

The main forward-pass reference is:

> **AI VIET NAM – AI Course 2025, _CNNs: Step-by-Step Examples_**  
> Nguyễn Phúc Thịnh and Đinh Quang Vinh

The reference explains CNN motivation, convolution, stride, padding, pooling, flattening, channels, and a manual CNN forward pass.

The **backward-pass derivations in this document are an added extension** to explain the calculations that PyTorch autograd performs automatically.

> **GitHub rendering note:** inline mathematics uses `$...$`, while multiline display equations use fenced `math` blocks for more reliable GitHub rendering.

---

## 1. From MLP to CNN

An MLP normally flattens an image before processing it:

```text
1 × 28 × 28
    ↓
Flatten
    ↓
784
    ↓
Fully Connected Layers
```

Flattening preserves the pixel values, but the next layer no longer receives them as an explicit 2D grid.

A CNN keeps the image as a spatial tensor:

```text
Channels × Height × Width
```

and processes local regions directly.

The main CNN ideas are:

```text
Local connectivity
        +
Weight sharing
        +
Spatial feature maps
        +
Hierarchical feature extraction
```

A CNN should therefore **not** be defined as:

```text
MLP + convolution preprocessing
```

Our particular model has:

```text
Convolutional feature extractor
        ↓
Flatten
        ↓
Linear classifier
```

but CNNs are defined by their convolutional processing, not by the presence of a fully connected classifier at the end.

### Architectural progression

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
    ↓
Classification
```

---

## 2. Baseline CNN Used in This Project

The notebook uses:

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
Linear(3136 → 10)
    ↓
10 logits
```

For a batch of size $N$, PyTorch stores images as:

```math
(N,\ C,\ H,\ W)
```

MNIST therefore enters the model as:

```math
(N,\ 1,\ 28,\ 28)
```

where:

- $N$: batch size,
- $C=1$: grayscale channel,
- $H=W=28$: image height and width.

### Shape tracking

| Stage | Output shape |
|---|---|
| Input | $N\times1\times28\times28$ |
| Conv1 | $N\times32\times28\times28$ |
| ReLU1 | $N\times32\times28\times28$ |
| Pool1 | $N\times32\times14\times14$ |
| Conv2 | $N\times64\times14\times14$ |
| ReLU2 | $N\times64\times14\times14$ |
| Pool2 | $N\times64\times7\times7$ |
| Flatten | $N\times3136$ |
| Linear | $N\times10$ |

---

## 3. Convolution — Forward Pass

### 3.1 Single-channel calculation

A convolutional layer uses a small learnable matrix called a **kernel** or **filter**.

For one input channel, stride $1$, and no padding:

```math
Y_{i,j}
=
\sum_{u=0}^{K_H-1}
\sum_{v=0}^{K_W-1}
W_{u,v}X_{i+u,j+v}
+b
```

where:

- $X$: input,
- $W$: kernel,
- $b$: bias,
- $Y$: output feature map.

Deep-learning libraries normally implement this as **cross-correlation**: the kernel is not flipped before the forward multiplication.

### 3.2 Manual example

Let:

```math
X=
\begin{bmatrix}
1&2&3\\
4&5&6\\
7&8&9
\end{bmatrix},
\qquad
W=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
```

with stride $1$, no padding, and no bias.

The top-left output is:

```math
Y_{0,0}
=
1(1)+2(0)+4(0)+5(1)
=
6
```

Moving the same kernel across the input gives:

```math
Y=
\begin{bmatrix}
6&8\\
12&14
\end{bmatrix}
```

The important point is that **the same kernel is reused at every position**.

That is **weight sharing**.

```text
Input patch
    ↓
Element-wise multiplication with kernel
    ↓
Sum
    ↓
Add bias
    ↓
One feature-map value
```

---

## 4. Stride, Padding, Channels, and Output Size

### 4.1 Stride

Stride controls how far the kernel moves.

```text
stride = 1
→ move one pixel

stride = 2
→ move two pixels
```

A larger stride usually produces a smaller output.

### 4.2 Padding

Padding adds values, usually zeros, around the input.

For example:

```text
4 × 4 input
    ↓ padding = 1
6 × 6 padded input
```

Padding can:

- preserve spatial size,
- allow border pixels to participate in more convolution windows,
- prevent feature maps from shrinking too quickly.

### 4.3 Output-size formula

For height:

```math
H_{\text{out}}
=
\left\lfloor
\frac{H_{\text{in}}+2P_H-K_H}{S_H}
\right\rfloor
+1
```

For width:

```math
W_{\text{out}}
=
\left\lfloor
\frac{W_{\text{in}}+2P_W-K_W}{S_W}
\right\rfloor
+1
```

For the first convolution:

```text
Input   = 28
Kernel  = 3
Padding = 1
Stride  = 1
```

so:

```math
H_{\text{out}}
=
\frac{28+2(1)-3}{1}+1
=
28
```

and:

```math
W_{\text{out}}=28
```

Therefore:

```text
1 × 28 × 28
      ↓
Conv2d(1 → 32, 3×3, padding=1)
      ↓
32 × 28 × 28
```

### 4.4 Multiple input channels

A grayscale image has:

```math
C_{\text{in}}=1
```

An RGB image has:

```math
C_{\text{in}}=3
```

A convolutional filter spans **all input channels**.

For a general multi-channel convolution:

```math
Y_{o,i,j}
=
b_o+
\sum_{c=1}^{C_{\text{in}}}
\sum_{u=0}^{K_H-1}
\sum_{v=0}^{K_W-1}
W_{o,c,u,v}X_{c,i+u,j+v}
```

where $o$ indexes the output channel.

One filter produces one output feature map.

Therefore:

```text
1 input channel
    ↓
32 filters
    ↓
32 output feature maps
```

For the second convolution:

```text
32 input channels
    ↓
64 filters
    ↓
64 output feature maps
```

Each of those 64 filters spans all 32 input channels.

---

## 5. Parameter Counting

For a standard `Conv2d` layer with bias:

```math
\text{parameters}
=
C_{\text{out}}
(C_{\text{in}}K_HK_W+1)
```

The $+1$ represents one bias per output channel.

### Conv1

```text
Conv2d(1 → 32, 3×3)
```

Weights:

```math
32\times1\times3\times3=288
```

Biases:

```math
32
```

Total:

```math
\boxed{320}
```

### Conv2

```text
Conv2d(32 → 64, 3×3)
```

Weights:

```math
64\times32\times3\times3=18,432
```

Biases:

```math
64
```

Total:

```math
\boxed{18,496}
```

### Linear classifier

The final feature vector has:

```math
64\times7\times7=3136
```

values.

For:

```text
Linear(3136 → 10)
```

the parameter count is:

```math
3136\times10+10
=
31,370
```

### Total

| Layer | Parameters |
| --- | ---: |
| Conv1 | 320 |
| Conv2 | 18,496 |
| Linear | 31,370 |
| **Total** | **50,186** |

ReLU, MaxPool, and Flatten have no trainable parameters.

A major advantage of convolution is that the kernel parameters are reused across the image. The parameter count therefore does not grow with the number of spatial positions at which a kernel is applied.

---

## 6. ReLU, Pooling, and Flatten

### 6.1 ReLU

ReLU is:

```math
\mathrm{ReLU}(x)=\max(0,x)
```

Example:

```math
[-2,\ 3,\ -1,\ 5]
\rightarrow
[0,\ 3,\ 0,\ 5]
```

Its derivative is:

```math
\mathrm{ReLU}'(x)
=
\begin{cases}
1,&x>0\\
0,&x\le0
\end{cases}
```

If the incoming gradient is $G$:

```math
\boxed{
\frac{\partial L}{\partial x}
=
G\odot\mathbf{1}[x>0]
}
```

where $\odot$ denotes element-wise multiplication.

At $x=0$, ReLU is not differentiable in the strict mathematical sense; PyTorch uses a gradient of zero there.

### 6.2 Max Pooling

For:

```math
A=
\begin{bmatrix}
5&2\\
3&7
\end{bmatrix}
```

a $2\times2$ MaxPool produces:

```math
P=7
```

The baseline uses:

```text
28 × 28
   ↓ MaxPool2d(2)
14 × 14

14 × 14
   ↓ MaxPool2d(2)
7 × 7
```

MaxPool has no learnable parameters.

#### MaxPool backward

Suppose:

```math
\frac{\partial L}{\partial P}=g
```

Because $7$ was the maximum:

```math
\boxed{
\frac{\partial L}{\partial A}
=
\begin{bmatrix}
0&0\\
0&g
\end{bmatrix}
}
```

The gradient is routed to the position selected during the forward pass.

If pooling windows overlap, contributions can accumulate.

### 6.3 Average Pooling

Average Pooling computes:

```math
P=
\frac{1}{K_HK_W}
\sum_{\text{window}}X
```

It is not used in this baseline, but it differs from MaxPool conceptually:

```text
MaxPool
→ keep the strongest response

AveragePool
→ keep the average response
```

### 6.4 Flatten

After the second pooling layer:

```math
64\times7\times7
\rightarrow
3136
```

Flatten changes only the shape.

Forward:

```text
64 × 7 × 7
    ↓
3136
```

Backward:

```text
3136-gradient
    ↓ reshape
64 × 7 × 7 gradient
```

Flatten has no parameters and performs no learned transformation.

---

## 7. Linear Classifier, Softmax, and Cross Entropy

### 7.1 Linear classifier

Let:

```math
f\in\mathbb{R}^{3136}
```

be the flattened CNN feature vector.

The final layer computes:

```math
z=Wf+b
```

with:

```math
W\in\mathbb{R}^{10\times3136},
\qquad
b\in\mathbb{R}^{10}
```

so:

```math
z\in\mathbb{R}^{10}
```

The ten values are **logits**, one for each MNIST class.

### 7.2 Softmax

Softmax converts logits into probabilities:

```math
p_k
=
\frac{e^{z_k}}
{\sum_j e^{z_j}}
```

with:

```math
\sum_k p_k=1
```

### 7.3 Cross Entropy

For one-hot target vector $y$:

```math
L
=
-\sum_k y_k\log p_k
```

If the correct class is $c$:

```math
L=-\log p_c
```

PyTorch's:

```python
nn.CrossEntropyLoss()
```

expects raw logits and internally performs the equivalent of log-Softmax followed by negative log-likelihood.

Therefore the notebook should use:

```python
logits = model(images)
loss = criterion(logits, labels)
```

and should **not** apply Softmax inside the model before `CrossEntropyLoss`.

### 7.4 Softmax + Cross Entropy gradient

For one sample:

```math
L
=
-\sum_k y_k z_k
+
\log\left(\sum_j e^{z_j}\right)
```

Differentiating with respect to $z_k$:

```math
\frac{\partial L}{\partial z_k}
=
-y_k
+
\frac{e^{z_k}}{\sum_j e^{z_j}}
```

Therefore:

```math
\boxed{
\frac{\partial L}{\partial z_k}
=
p_k-y_k
}
```

This is the gradient that starts the backward pass.

With PyTorch's default `reduction="mean"` over a batch of $N$:

```math
\boxed{
\frac{\partial L}{\partial z_{n,k}}
=
\frac{p_{n,k}-y_{n,k}}{N}
}
```

for the standard one-hot interpretation of the labels.

---

## 8. Backpropagation Through the CNN

The notebook contains:

```python
loss.backward()
```

That line represents the chain:

```text
Loss
 ↓
Softmax + Cross Entropy
 ↓
Logits
 ↓
Linear
 ↓
Flatten
 ↓
MaxPool2
 ↓
ReLU2
 ↓
Conv2
 ↓
MaxPool1
 ↓
ReLU1
 ↓
Conv1
```

Backpropagation repeatedly applies the chain rule.

### 8.1 Linear layer backward

For one sample:

```math
z=Wf+b
```

Let:

```math
g_z=\frac{\partial L}{\partial z}
```

Then:

```math
\boxed{
\frac{\partial L}{\partial W}
=
g_zf^T
}
```

```math
\boxed{
\frac{\partial L}{\partial b}
=
g_z
}
```

and the gradient passed to the previous layer is:

```math
\boxed{
\frac{\partial L}{\partial f}
=
W^Tg_z
}
```

For a batch with:

```math
F\in\mathbb{R}^{N\times D},
\qquad
G_Z\in\mathbb{R}^{N\times C}
```

the weight gradient is:

```math
\boxed{
\frac{\partial L}{\partial W}
=
G_Z^TF
}
```

with any averaging factor already contained in $G_Z$.

### 8.2 Flatten backward

Flatten only restores the gradient to its previous shape:

```text
N × 3136
    ↓ reshape
N × 64 × 7 × 7
```

### 8.3 MaxPool backward

Each pooled output sends its gradient to the input position that produced the selected maximum.

All other positions in that pooling window receive zero from that output.

### 8.4 ReLU backward

If:

```math
A=\mathrm{ReLU}(Y)
```

and:

```math
G_A=\frac{\partial L}{\partial A}
```

then:

```math
\boxed{
G_Y
=
G_A\odot\mathbf{1}[Y>0]
}
```

### 8.5 Convolution backward

For readability, first assume:

- one input channel,
- one output channel,
- stride $1$,
- no padding.

The forward operation is:

```math
Y_{i,j}
=
\sum_{u,v}W_{u,v}X_{i+u,j+v}+b
```

Let:

```math
G_{i,j}
=
\frac{\partial L}{\partial Y_{i,j}}
```

We need gradients for:

```text
kernel W
bias b
input X
```

#### Kernel gradient

Because:

```math
\frac{\partial Y_{i,j}}{\partial W_{u,v}}
=
X_{i+u,j+v}
```

the chain rule gives:

```math
\boxed{
\frac{\partial L}{\partial W_{u,v}}
=
\sum_{i,j}
G_{i,j}X_{i+u,j+v}
}
```

This is the key mathematical effect of **weight sharing**:

> the same kernel weight is used at many positions, so its gradient accumulates contributions from every position where it was used.

#### Bias gradient

Because the same bias is added at every output position:

```math
\boxed{
\frac{\partial L}{\partial b}
=
\sum_{i,j}G_{i,j}
}
```

#### Input gradient

An input value can participate in several overlapping convolution windows.

The most readable way to compute its gradient is a **scatter-add** view:

```text
For every output position (i, j):

1. take G[i, j]
2. multiply the kernel by G[i, j]
3. add that matrix into the input-gradient region
   that produced Y[i, j]
```

Therefore each input value receives the sum of all gradient contributions from output values that depended on it.

### 8.6 Multi-channel convolution gradients

For a batch:

```math
Y_{n,o,i,j}
=
b_o+
\sum_c\sum_u\sum_v
W_{o,c,u,v}
X_{n,c,i+u,j+v}
```

the weight gradient is:

```math
\boxed{
\frac{\partial L}{\partial W_{o,c,u,v}}
=
\sum_{n,i,j}
G_{n,o,i,j}
X_{n,c,i+u,j+v}
}
```

and:

```math
\boxed{
\frac{\partial L}{\partial b_o}
=
\sum_{n,i,j}
G_{n,o,i,j}
}
```

The input gradient for channel $c$ accumulates contributions from every output channel whose filters used that input channel.

For the notebook's second convolution:

```text
32 input channels
    ↓
64 filters
    ↓
64 output channels
```

each of the 32 input feature maps receives gradient contributions from all relevant output filters.

---

## 9. Complete Numerical Forward + Backward Example

This compact example follows the same layer order as the real network:

```text
Conv
 ↓
ReLU
 ↓
MaxPool
 ↓
Linear
 ↓
Cross Entropy
```

The dimensions are deliberately tiny so every calculation can be followed manually.

### 9.1 Setup

Input:

```math
X=
\begin{bmatrix}
1&2&0\\
0&1&3\\
2&1&0
\end{bmatrix}
```

Kernel:

```math
W=
\begin{bmatrix}
1&-1\\
0&2
\end{bmatrix}
```

Convolution bias:

```math
b=0
```

Use:

```text
stride = 1
padding = 0
```

The convolution output has size:

```math
\frac{3-2}{1}+1=2
```

so $Y$ is $2\times2$.

### 9.2 Convolution forward

The four output values are:

```math
Y_{0,0}
=
1(1)+2(-1)+0(0)+1(2)
=
1
```

```math
Y_{0,1}
=
2(1)+0(-1)+1(0)+3(2)
=
8
```

```math
Y_{1,0}
=
0(1)+1(-1)+2(0)+1(2)
=
1
```

```math
Y_{1,1}
=
1(1)+3(-1)+1(0)+0(2)
=
-2
```

Therefore:

```math
Y=
\begin{bmatrix}
1&8\\
1&-2
\end{bmatrix}
```

### 9.3 ReLU and MaxPool forward

ReLU gives:

```math
A=
\begin{bmatrix}
1&8\\
1&0
\end{bmatrix}
```

A $2\times2$ MaxPool gives:

```math
f=\max(1,8,1,0)=8
```

The maximum came from position $(0,1)$.

### 9.4 Linear classifier

Use two output classes:

```math
W_{\text{fc}}
=
\begin{bmatrix}
0.2\\
-0.1
\end{bmatrix},
\qquad
b_{\text{fc}}
=
\begin{bmatrix}
0\\
0
\end{bmatrix}
```

Then:

```math
z=W_{\text{fc}}f+b_{\text{fc}}
=
\begin{bmatrix}
1.6\\
-0.8
\end{bmatrix}
```

Assume class 0 is correct:

```math
y=
\begin{bmatrix}
1\\
0
\end{bmatrix}
```

### 9.5 Softmax and loss

```math
p_0
=
\frac{e^{1.6}}{e^{1.6}+e^{-0.8}}
\approx
0.9168
```

```math
p_1\approx0.0832
```

Therefore:

```math
p\approx
\begin{bmatrix}
0.9168\\
0.0832
\end{bmatrix}
```

and:

```math
L=-\log(0.9168)\approx0.0868
```

The forward pass is complete.

---

### 9.6 Start the backward pass

For Softmax + Cross Entropy:

```math
g_z
=
p-y
```

so:

```math
g_z
\approx
\begin{bmatrix}
-0.0832\\
0.0832
\end{bmatrix}
```

### 9.7 Linear-layer gradients

Weight gradient:

```math
\frac{\partial L}{\partial W_{\text{fc}}}
=
g_zf
```

With $f=8$:

```math
\boxed{
\frac{\partial L}{\partial W_{\text{fc}}}
\approx
\begin{bmatrix}
-0.6654\\
0.6654
\end{bmatrix}
}
```

Bias gradient:

```math
\boxed{
\frac{\partial L}{\partial b_{\text{fc}}}
\approx
\begin{bmatrix}
-0.0832\\
0.0832
\end{bmatrix}
}
```

Gradient with respect to the pooled feature:

```math
\frac{\partial L}{\partial f}
=
W_{\text{fc}}^Tg_z
```

```math
=
0.2(-0.0832)+(-0.1)(0.0832)
\approx
-0.0250
```

### 9.8 MaxPool backward

The forward maximum came from $A_{0,1}=8$.

Therefore:

```math
\frac{\partial L}{\partial A}
\approx
\begin{bmatrix}
0&-0.0250\\
0&0
\end{bmatrix}
```

### 9.9 ReLU backward

The convolution output was:

```math
Y=
\begin{bmatrix}
1&8\\
1&-2
\end{bmatrix}
```

so the ReLU mask is:

```math
\mathbf{1}[Y>0]
=
\begin{bmatrix}
1&1\\
1&0
\end{bmatrix}
```

Hence:

```math
\frac{\partial L}{\partial Y}
=
\frac{\partial L}{\partial A}
\odot
\mathbf{1}[Y>0]
\approx
\begin{bmatrix}
0&-0.0250\\
0&0
\end{bmatrix}
```

Only $Y_{0,1}$ carries a nonzero gradient.

### 9.10 Kernel gradient

The input patch that produced $Y_{0,1}$ was:

```math
\begin{bmatrix}
2&0\\
1&3
\end{bmatrix}
```

Therefore:

```math
\frac{\partial L}{\partial W}
=
(-0.0250)
\begin{bmatrix}
2&0\\
1&3
\end{bmatrix}
```

so:

```math
\boxed{
\frac{\partial L}{\partial W}
\approx
\begin{bmatrix}
-0.0499&0\\
-0.0250&-0.0749
\end{bmatrix}
}
```

If several convolution outputs had nonzero gradients, their patch contributions would be added together.

### 9.11 Convolution bias gradient

Only one convolution output has a nonzero gradient, so:

```math
\boxed{
\frac{\partial L}{\partial b}
\approx
-0.0250
}
```

### 9.12 Input gradient

The nonzero gradient came from output position $(0,1)$, whose input patch used:

```text
rows    0..1
columns 1..2
```

Multiply the kernel by the incoming gradient:

```math
(-0.0250)
\begin{bmatrix}
1&-1\\
0&2
\end{bmatrix}
=
\begin{bmatrix}
-0.0250&0.0250\\
0&-0.0499
\end{bmatrix}
```

Scatter this contribution back into the matching input region:

```math
\boxed{
\frac{\partial L}{\partial X}
\approx
\begin{bmatrix}
0&-0.0250&0.0250\\
0&0&-0.0499\\
0&0&0
\end{bmatrix}
}
```

With several nonzero output gradients, overlapping contributions would be added.

---

## 10. What `loss.backward()` Computes in the Real CNN

For the actual MNIST model:

```text
CrossEntropyLoss
      ↓
fc.weight.grad       : 10 × 3136
fc.bias.grad         : 10
      ↓
reshape
      ↓
Pool2 backward
      ↓
ReLU2 backward
      ↓
conv2.weight.grad    : 64 × 32 × 3 × 3
conv2.bias.grad      : 64
      ↓
Pool1 backward
      ↓
ReLU1 backward
      ↓
conv1.weight.grad    : 32 × 1 × 3 × 3
conv1.bias.grad      : 32
```

For a batch, each parameter gradient combines contributions from all examples.

With `CrossEntropyLoss()` using its default mean reduction, the per-example contributions are averaged through the loss gradient.

---

## 11. SGD Update

After:

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

the important steps are:

```text
optimizer.zero_grad()
→ clear gradients from the previous batch

loss.backward()
→ compute current gradients

optimizer.step()
→ update parameters
```

For a parameter $\theta$, basic SGD performs:

```math
\boxed{
\theta
\leftarrow
\theta
-
\eta
\frac{\partial L}{\partial\theta}
}
```

where $\eta$ is the learning rate.

This completes one training step.

---

## 12. Why Convolution Learns Useful Features

Convolution kernels begin as learned parameters with initial values; they do not start as hand-written edge or curve detectors.

Training repeatedly performs:

```text
Kernel
 ↓
Feature map
 ↓
Later layers
 ↓
Logits
 ↓
Cross Entropy
 ↓
Gradient
 ↓
Kernel update
```

If changing a kernel weight would reduce the loss, gradient descent moves that weight in the corresponding direction.

Because one kernel is reused across many spatial positions:

```math
\frac{\partial L}{\partial W_{u,v}}
=
\sum_{i,j}
G_{i,j}X_{i+u,j+v}
```

one shared parameter can learn from many image locations.

This gives convolution two important properties:

1. **Parameter efficiency**  
   The network does not need a different local detector at every position.

2. **Feature reuse across location**  
   A local pattern learned in one region can also be detected elsewhere.

### Receptive field intuition

A first-layer convolution sees only a local region.

As layers are stacked, later features depend on increasingly large regions of the original image.

For this baseline:

| Stage | Receptive field | Effective jump |
| --- | ---: |---:|
| Input | $1\times1$ | 1 |
| Conv1 $3\times3,\ s=1$ | $3\times3$ | 1 |
| Pool1 $2\times2,\ s=2$ | $4\times4$ | 2 |
| Conv2 $3\times3,\ s=1$ | $8\times8$ | 2 |
| Pool2 $2\times2,\ s=2$ | $10\times10$ | 4 |

This is how local processing becomes a hierarchical spatial representation.

---

## 13. What Changed from MLP?

### Reused

```text
MNIST
Mini-batches
ReLU
Logits
Cross Entropy
Backpropagation
SGD
Train / validation / test
Accuracy
```

The final linear classifier also uses the same mathematics already seen in the MLP stage.

### New in CNN

```text
Spatial tensors
Local connectivity
Kernels / filters
Weight sharing
Feature maps
Channels
Stride
Padding
Pooling
Hierarchical spatial representation
Convolution-specific gradients
```

The main change is the **representation and feature-processing stage**:

```text
MLP
Raw image
 ↓
Flatten
 ↓
Fully connected representation

CNN
Raw image
 ↓
Local convolutional processing
 ↓
Spatial feature maps
 ↓
Hierarchical representation
 ↓
Classification
```

---

## 14. Notebook vs Documentation

### `notebooks/05_cnn.ipynb`

Use the notebook to understand:

```text
How to implement the CNN
How tensor shapes change
How to train it in PyTorch
How to evaluate it
How learned feature maps look
```

It intentionally lets PyTorch autograd handle the derivatives.

### `docs/05_cnn.md`

Use this document to understand:

```text
What each operation calculates
Why the tensor shapes change
How many parameters are learned
How Cross Entropy creates the first gradient
How gradients pass through Linear
How gradients pass through Flatten
How gradients pass through MaxPool
How gradients pass through ReLU
How convolution kernel / bias / input gradients are formed
Why shared filters accumulate gradient from many positions
```

Together:

```text
Notebook
= implementation intuition

Documentation
= mathematical intuition
```

---

## 15. Key Takeaways

1. CNNs process images with spatially structured operations instead of flattening immediately.

2. Convolution uses local connectivity and reuses the same kernel across spatial positions.

3. One filter spans all input channels and produces one output feature map.

4. Kernel size, stride, and padding determine spatial output dimensions.

5. ReLU introduces nonlinearity and blocks gradients where its input is non-positive.

6. MaxPool keeps a selected maximum during the forward pass and routes its backward gradient to that selected position.

7. Flatten only reshapes values; its backward pass reshapes the gradient back.

8. The final linear classifier uses the same core mathematics as an MLP output layer.

9. For Softmax + Cross Entropy:

```math
\boxed{
\frac{\partial L}{\partial z}=p-y
}
```

for one sample.

10. For convolution:

```math
\boxed{
\frac{\partial L}{\partial W_{u,v}}
=
\sum_{i,j}
G_{i,j}X_{i+u,j+v}
}
```

so shared weights accumulate gradient contributions from all positions where they are used.

11. `loss.backward()` performs the complete backward chain automatically.

12. The baseline CNN contains:

```math
\boxed{50,186}
```

trainable parameters.

---

## Reference

**AI VIET NAM – AI Course 2025**  
**_CNNs: Step-by-Step Examples_**  
Nguyễn Phúc Thịnh and Đinh Quang Vinh

Reference topics used here:

- motivation for CNNs,
- convolution,
- stride,
- padding,
- pooling,
- flattening,
- image channels,
- and manual forward-pass calculations.

The backward-pass derivations and complete numerical backward example are companion material added to explain the operations hidden by PyTorch autograd.
