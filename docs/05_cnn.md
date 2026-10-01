# Convolutional Neural Network (CNN)

## Purpose

This document is the **theory and mathematics companion** to `notebooks/05_cnn.ipynb`.

The notebook is intentionally simple:

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

It focuses on understanding the architecture and keeping the implementation readable.

This document goes deeper into what the notebook leaves to PyTorch:

- the mathematics of convolution,
- stride, padding, channels, and output shapes,
- parameter counting,
- ReLU and pooling calculations,
- the final linear classifier,
- Softmax + Cross Entropy,
- and especially the **backward pass through a CNN**.

The primary forward-pass reference is:

> **AI VIET NAM – AI Course 2025, _CNNs: Step-by-Step Examples_**  
> Nguyễn Phúc Thịnh and Đinh Quang Vinh

That reading explains CNN motivation, convolution, stride, padding, pooling, flattening, channels, and a manual CNN forward pass.

The **backward-pass derivations in this document are an added extension**. The reference reading focuses on the forward calculations rather than deriving convolution gradients.

---

# 1. From MLP to CNN

The previous MLP receives an MNIST image and immediately flattens it:

```text
1 × 28 × 28
    ↓
Flatten
    ↓
784
    ↓
Fully Connected Layers
```

Flattening does not destroy pixel values, but it removes the image's **explicit 2D organization** from the representation used by the next layer.

A CNN instead keeps the image as a spatial tensor:

```text
Channels × Height × Width
```

and performs operations directly on local image regions.

The important CNN ideas are:

```text
Local connectivity
        +
Weight sharing
        +
Spatial feature maps
        +
Hierarchical feature extraction
```

A CNN should therefore **not** be defined as merely:

```text
MLP + convolution preprocessing
```

Our particular model does have:

```text
Convolutional feature extractor
        ↓
Flatten
        ↓
Linear classifier
```

but the defining idea of a CNN is the use of convolutional processing with local connectivity and shared parameters.

Other CNNs may use:

- deeper convolutional blocks,
- global average pooling,
- convolutional classification heads,
- residual blocks,
- or other structures.

---

# 2. CNN Architecture Used in This Project

The notebook uses the following baseline:

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

For a batch of size \(N\), PyTorch stores image tensors as:

\[
(N,\ C,\ H,\ W)
\]

so MNIST enters the model as:

\[
(N,\ 1,\ 28,\ 28)
\]

where:

- \(N\) = batch size,
- \(C=1\) = grayscale channel,
- \(H=W=28\).

---

# 3. Convolution — Forward Pass

## 3.1 Local calculation

A convolutional layer uses a small learnable matrix called a **kernel** or **filter**.

For a single input channel, one output value is calculated from a local patch:

\[
Y_{i,j}
=
\sum_{u=0}^{K_H-1}
\sum_{v=0}^{K_W-1}
W_{u,v}
X_{i+u,j+v}
+b
\]

where:

- \(X\) = input,
- \(W\) = kernel weights,
- \(b\) = bias,
- \(Y\) = output feature map.

In deep-learning libraries, the operation normally called "convolution" is technically **cross-correlation** because the kernel is not flipped before multiplication.

That is also the convention used by PyTorch `nn.Conv2d`.

---

## 3.2 Small manual example

The reference reading uses:

\[
X=
\begin{bmatrix}
1&2&3\\
4&5&6\\
7&8&9
\end{bmatrix}
\]

and

\[
W=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}
\]

with no padding, stride \(1\), and no bias.

At the top-left position:

\[
Y_{0,0}
=
1(1)+2(0)+4(0)+5(1)
=6
\]

Move the kernel one position right:

\[
Y_{0,1}
=
2(1)+3(0)+5(0)+6(1)
=8
\]

Move down:

\[
Y_{1,0}
=
4(1)+5(0)+7(0)+8(1)
=12
\]

and:

\[
Y_{1,1}
=
5(1)+6(0)+8(0)+9(1)
=14
\]

Therefore:

\[
Y=
\begin{bmatrix}
6&8\\
12&14
\end{bmatrix}
\]

This is the essential convolution calculation:

```text
Select local patch
      ↓
Element-wise multiply with kernel
      ↓
Sum
      ↓
Add bias
      ↓
One feature-map value
```

The same kernel is then reused at every spatial position.

That reuse is **weight sharing**.

---

# 4. Stride, Padding, and Output Size

## 4.1 Stride

Stride determines how far the kernel moves.

```text
stride = 1
→ move one pixel

stride = 2
→ move two pixels
```

A larger stride usually produces a smaller output.

---

## 4.2 Padding

Padding adds values, normally zeros, around the input.

For example:

```text
Original 4 × 4
      ↓
padding = 1
      ↓
Padded 6 × 6
```

Padding is useful because it can:

- preserve spatial size,
- allow border pixels to participate in more convolution windows,
- prevent feature maps from shrinking too quickly.

---

## 4.3 Output-size formula

For height:

\[
H_{\text{out}}
=
\left\lfloor
\frac{H_{\text{in}}+2P_H-K_H}{S_H}
\right\rfloor
+1
\]

For width:

\[
W_{\text{out}}
=
\left\lfloor
\frac{W_{\text{in}}+2P_W-K_W}{S_W}
\right\rfloor
+1
\]

For the first convolution in our model:

```text
Input   = 28
Kernel  = 3
Padding = 1
Stride  = 1
```

so:

\[
H_{\text{out}}
=
\frac{28+2(1)-3}{1}+1
=
28
\]

and similarly:

\[
W_{\text{out}}=28
\]

Therefore:

```text
1 × 28 × 28
      ↓
Conv2d(1 → 32, 3×3, padding=1)
      ↓
32 × 28 × 28
```

---

# 5. Channels and Multiple Filters

## 5.1 Input channels

A grayscale MNIST image has:

\[
C_{\text{in}}=1
\]

An RGB image has:

\[
C_{\text{in}}=3
\]

A convolutional kernel always spans **all input channels**.

If the input has 3 channels and the spatial kernel is \(3\times3\), one filter has shape:

\[
3\times3\times3
\]

or in PyTorch order:

```text
Cin × KH × KW
```

---

## 5.2 General multi-channel convolution

For output channel \(o\):

\[
Y_{o,i,j}
=
b_o
+
\sum_{c=1}^{C_{\text{in}}}
\sum_{u=0}^{K_H-1}
\sum_{v=0}^{K_W-1}
W_{o,c,u,v}
X_{c,i+u,j+v}
\]

One filter therefore combines information across **all input channels** and produces **one output feature map**.

If we want 32 output channels, we need 32 different filters.

Thus:

```text
1 input channel
      ↓
32 different filters
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

# 6. Convolution Parameter Count

For a standard `Conv2d` layer:

\[
\text{weights}
=
C_{\text{out}}
\times
C_{\text{in}}
\times
K_H
\times
K_W
\]

and if bias is enabled:

\[
\text{biases}=C_{\text{out}}
\]

so:

\[
\boxed{
\text{parameters}
=
C_{\text{out}}
(C_{\text{in}}K_HK_W+1)
}
\]

---

## 6.1 First convolution

```text
Conv2d(1 → 32, 3×3)
```

Weights:

\[
32\times1\times3\times3
=
288
\]

Biases:

\[
32
\]

Total:

\[
\boxed{320}
\]

---

## 6.2 Second convolution

```text
Conv2d(32 → 64, 3×3)
```

Weights:

\[
64\times32\times3\times3
=
18,432
\]

Biases:

\[
64
\]

Total:

\[
\boxed{18,496}
\]

Notice that the same \(3\times3\) weights are reused across the whole image.

The number of parameters does **not** depend on the image width and height.

That is one major advantage of parameter sharing.

---

# 7. ReLU

After convolution, the notebook applies:

\[
\operatorname{ReLU}(x)=\max(0,x)
\]

Example:

\[
[-2,\ 3,\ -1,\ 5]
\]

becomes:

\[
[0,\ 3,\ 0,\ 5]
\]

ReLU adds nonlinearity.

Without nonlinear activations, stacking multiple linear operations would still collapse into an overall linear transformation.

---

## 7.1 ReLU derivative

The derivative used during backpropagation is:

\[
\operatorname{ReLU}'(x)
=
\begin{cases}
1,&x>0\\
0,&x\le0
\end{cases}
\]

Therefore, if the incoming gradient is \(G\):

\[
\boxed{
\frac{\partial L}{\partial x}
=
G\cdot \mathbf{1}[x>0]
}
\]

Interpretation:

```text
Positive pre-activation
→ gradient passes through

Negative pre-activation
→ gradient becomes zero
```

At exactly \(x=0\), ReLU is not mathematically differentiable; implementations use a chosen subgradient convention. PyTorch uses zero there.

---

# 8. Pooling

Pooling reduces spatial resolution.

The notebook uses:

```text
MaxPool2d(kernel_size=2)
```

With the default stride equal to the kernel size, a \(2\times2\) window reduces:

```text
28 × 28
   ↓
14 × 14
```

and later:

```text
14 × 14
   ↓
7 × 7
```

Pooling has **no learnable weights**.

---

## 8.1 Max Pooling forward calculation

Suppose a pooling window is:

\[
\begin{bmatrix}
5&2\\
3&7
\end{bmatrix}
\]

Then:

\[
\max(5,2,3,7)=7
\]

so the pooled output is:

\[
7
\]

The reference reading uses this same idea to reduce feature maps while retaining the strongest response in each region.

---

## 8.2 Average Pooling

Average Pooling instead computes:

\[
P
=
\frac{1}{K_HK_W}
\sum_{\text{window}} X
\]

For:

\[
\begin{bmatrix}
2&4\\
6&8
\end{bmatrix}
\]

the result is:

\[
\frac{2+4+6+8}{4}=5
\]

Our baseline does not use Average Pooling, but the distinction is useful:

```text
Max Pooling
→ preserve strongest response

Average Pooling
→ preserve average response
```

---

# 9. Flatten

After the second pooling layer, the tensor has shape:

\[
64\times7\times7
\]

Flatten converts it into:

\[
64\cdot7\cdot7=3136
\]

features.

```text
64 × 7 × 7
    ↓
Flatten
    ↓
3136
```

Flatten has:

\[
\boxed{0\text{ trainable parameters}}
\]

It only changes the tensor's shape.

---

# 10. Linear Classifier

The final classifier is:

```python
nn.Linear(64 * 7 * 7, 10)
```

For one example, let:

\[
f\in\mathbb{R}^{3136}
\]

be the flattened CNN features.

The logits are:

\[
z=Wf+b
\]

where:

\[
W\in\mathbb{R}^{10\times3136}
\]

and:

\[
b\in\mathbb{R}^{10}
\]

Therefore:

\[
z\in\mathbb{R}^{10}
\]

One logit is produced for each MNIST class.

---

## 10.1 Linear-layer parameter count

Weights:

\[
10\times3136
=
31,360
\]

Biases:

\[
10
\]

Total:

\[
\boxed{31,370}
\]

---

# 11. Total Parameters in the Baseline CNN

The complete model has:

| Layer | Parameters |
|---|---:|
| Conv1 | 320 |
| Conv2 | 18,496 |
| Linear | 31,370 |
| **Total** | **50,186** |

Therefore:

\[
\boxed{
\text{Total trainable parameters}=50,186
}
\]

Pooling, ReLU, and Flatten contribute no trainable parameters.

---

# 12. Forward Shape Tracking

For batch size \(N\):

```text
Input
N × 1 × 28 × 28

↓ Conv1

N × 32 × 28 × 28

↓ ReLU

N × 32 × 28 × 28

↓ Pool1

N × 32 × 14 × 14

↓ Conv2

N × 64 × 14 × 14

↓ ReLU

N × 64 × 14 × 14

↓ Pool2

N × 64 × 7 × 7

↓ Flatten

N × 3136

↓ Linear

N × 10
```

This shape sequence is exactly what the notebook implements.

---

# 13. Softmax and Cross Entropy

The model outputs **logits**, not probabilities.

For one example:

\[
z=
[z_1,z_2,\ldots,z_{10}]
\]

Softmax converts logits to probabilities:

\[
p_k
=
\frac{e^{z_k}}
{\sum_j e^{z_j}}
\]

so:

\[
\sum_k p_k=1
\]

For one-hot target vector \(y\), Cross Entropy is:

\[
L
=
-\sum_k y_k\log p_k
\]

If the correct class is \(c\):

\[
L=-\log p_c
\]

PyTorch's:

```python
nn.CrossEntropyLoss()
```

combines the equivalent log-Softmax and negative log-likelihood operations internally.

Therefore the notebook correctly sends raw logits directly into the loss:

```python
logits = model(images)
loss = criterion(logits, labels)
```

No explicit Softmax should be placed in the model during training.

---

# 14. The Important Softmax + Cross Entropy Gradient

This derivative begins the backward pass.

Starting from:

\[
p_k
=
\frac{e^{z_k}}
{\sum_j e^{z_j}}
\]

and:

\[
L=-\sum_k y_k\log p_k
\]

we can rewrite the loss as:

\[
L
=
-\sum_k y_k z_k
+
\log\left(\sum_j e^{z_j}\right)
\]

for a one-hot target whose entries sum to 1.

Differentiate with respect to logit \(z_k\):

\[
\frac{\partial L}{\partial z_k}
=
-y_k
+
\frac{e^{z_k}}
{\sum_j e^{z_j}}
\]

therefore:

\[
\boxed{
\frac{\partial L}{\partial z_k}
=
p_k-y_k
}
\]

This result is extremely important.

For the correct class:

```text
y = 1
→ gradient = predicted_probability - 1
```

For every incorrect class:

```text
y = 0
→ gradient = predicted_probability
```

With PyTorch's default `reduction="mean"` over a batch of \(N\) samples:

\[
\boxed{
\frac{\partial L}{\partial z_{n,k}}
=
\frac{p_{n,k}-y_{n,k}}{N}
}
\]

The division by \(N\) comes from averaging the per-sample losses.

---

# 15. Backpropagation Through the CNN

The notebook contains:

```python
loss.backward()
```

That one line hides the following chain:

```text
Loss
 ↓
Softmax + Cross Entropy
 ↓
10 logits
 ↓
Linear classifier
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
 ↓
earlier inputs / parameters
```

Backpropagation repeatedly applies the chain rule.

If:

\[
L=L(a),\quad a=a(b),\quad b=b(c)
\]

then:

\[
\frac{\partial L}{\partial c}
=
\frac{\partial L}{\partial a}
\frac{\partial a}{\partial b}
\frac{\partial b}{\partial c}
\]

Each layer receives an incoming gradient, computes the gradients needed for its own parameters, and passes another gradient to the previous layer.

---

# 16. Linear Layer — Backward Pass

For one sample:

\[
z=Wf+b
\]

Let:

\[
g_z
=
\frac{\partial L}{\partial z}
\]

Then:

## 16.1 Gradient with respect to weights

Each weight \(W_{k,j}\) contributes:

\[
z_k
=
\sum_j W_{k,j}f_j+b_k
\]

so:

\[
\frac{\partial z_k}{\partial W_{k,j}}
=
f_j
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial W}
=
g_z f^T
}
\]

For a batch represented as:

\[
F\in\mathbb{R}^{N\times D}
\]

and:

\[
G_Z\in\mathbb{R}^{N\times C}
\]

PyTorch-style dimensions give:

\[
\boxed{
\frac{\partial L}{\partial W}
=
G_Z^T F
}
\]

where:

\[
W\in\mathbb{R}^{C\times D}
\]

---

## 16.2 Gradient with respect to bias

Because:

\[
\frac{\partial z_k}{\partial b_k}=1
\]

for one sample:

\[
\boxed{
\frac{\partial L}{\partial b}
=
g_z
}
\]

For a batch:

\[
\boxed{
\frac{\partial L}{\partial b}
=
\sum_{n=1}^{N}G_{Z,n}
}
\]

with any batch averaging already included in \(G_Z\).

---

## 16.3 Gradient with respect to input features

To continue backward:

\[
\boxed{
\frac{\partial L}{\partial f}
=
W^Tg_z
}
\]

This gradient is then passed to Flatten.

---

# 17. Flatten — Backward Pass

Forward:

```text
64 × 7 × 7
    ↓
3136
```

Flatten does not change values, only their arrangement.

Therefore backward simply reverses the reshape:

```text
gradient shape:
3136
  ↓ reshape
64 × 7 × 7
```

If:

\[
g_f
=
\frac{\partial L}{\partial f}
\]

then the gradient before Flatten contains the exact same numbers, rearranged into the original tensor shape.

Flatten introduces:

\[
\boxed{0\text{ additional arithmetic gradients}}
\]

and:

\[
\boxed{0\text{ parameter gradients}}
\]

---

# 18. Max Pooling — Backward Pass

Consider:

\[
A=
\begin{bmatrix}
2&7\\
4&3
\end{bmatrix}
\]

Forward MaxPool gives:

\[
P=7
\]

because \(7\) is the maximum.

Suppose the gradient arriving from the next layer is:

\[
\frac{\partial L}{\partial P}=g
\]

Only the location that produced the maximum receives that gradient:

\[
\boxed{
\frac{\partial L}{\partial A}
=
\begin{bmatrix}
0&g\\
0&0
\end{bmatrix}
}
\]

Conceptually:

```text
Forward:
remember which element won the maximum

Backward:
send the gradient only to that element
```

PyTorch stores the maximum locations needed for this backward routing.

If pooling windows overlap, gradient contributions may accumulate at an input location.

MaxPool contains no learned weights, so there is no:

\[
\frac{\partial L}{\partial W_{\text{pool}}}
\]

to calculate.

---

# 19. ReLU — Backward Pass

Forward:

\[
A=\operatorname{ReLU}(Y)
\]

Suppose:

\[
G_A
=
\frac{\partial L}{\partial A}
\]

Then:

\[
\boxed{
G_Y
=
G_A
\odot
\mathbf{1}[Y>0]
}
\]

where \(\odot\) means element-wise multiplication.

Example:

\[
Y=
\begin{bmatrix}
2&-3\\
4&-1
\end{bmatrix}
\]

and incoming gradient:

\[
G_A=
\begin{bmatrix}
5&6\\
7&8
\end{bmatrix}
\]

The derivative mask is:

\[
\mathbf{1}[Y>0]
=
\begin{bmatrix}
1&0\\
1&0
\end{bmatrix}
\]

Therefore:

\[
G_Y=
\begin{bmatrix}
5&0\\
7&0
\end{bmatrix}
\]

Negative ReLU inputs block the gradient.

---

# 20. Convolution — Backward Pass

This is the main new gradient calculation introduced by CNNs.

For clarity, first consider one input channel and one output channel with stride \(1\) and no padding:

\[
Y_{i,j}
=
\sum_{u,v}
W_{u,v}X_{i+u,j+v}
+b
\]

Let:

\[
G_{i,j}
=
\frac{\partial L}{\partial Y_{i,j}}
\]

be the gradient arriving from the next layer.

We need:

```text
1. dL/dW   → how should the kernel change?
2. dL/db   → how should the bias change?
3. dL/dX   → what gradient should be passed to the previous layer?
```

---

# 21. Gradient with Respect to the Convolution Kernel

For a single kernel element \(W_{u,v}\):

\[
Y_{i,j}
=
\cdots
+
W_{u,v}X_{i+u,j+v}
+\cdots
\]

so:

\[
\frac{\partial Y_{i,j}}
{\partial W_{u,v}}
=
X_{i+u,j+v}
\]

By the chain rule:

\[
\frac{\partial L}
{\partial W_{u,v}}
=
\sum_{i,j}
\frac{\partial L}{\partial Y_{i,j}}
\frac{\partial Y_{i,j}}{\partial W_{u,v}}
\]

Therefore:

\[
\boxed{
\frac{\partial L}
{\partial W_{u,v}}
=
\sum_{i,j}
G_{i,j}
X_{i+u,j+v}
}
\]

This equation is one of the most important mathematical consequences of **weight sharing**.

The same kernel weight is reused at many image locations.

Therefore its final gradient is the **sum of contributions from every location where that weight was used**.

---

# 22. Gradient with Respect to Convolution Bias

Bias is added to every output position:

\[
Y_{i,j}
=
\cdots+b
\]

Therefore:

\[
\frac{\partial Y_{i,j}}{\partial b}=1
\]

and:

\[
\boxed{
\frac{\partial L}{\partial b}
=
\sum_{i,j}G_{i,j}
}
\]

For multiple samples in a batch, the sum also extends over the batch dimension.

For multiple output channels, each output channel has its own bias.

---

# 23. Gradient with Respect to the Convolution Input

An input value can influence several overlapping convolution windows.

Therefore its gradient must accumulate contributions from every output value that used it.

A convenient algorithmic interpretation is:

For every output position \((i,j)\):

```text
incoming gradient = G[i,j]

take the kernel W

multiply W by G[i,j]

add that result into the corresponding input window
```

Symbolically:

\[
\boxed{
\frac{\partial L}{\partial X}
=
\text{accumulated contributions from }G\text{ weighted by }W
}
\]

For stride \(1\), this accumulation resembles a convolution of the output gradient with a spatially flipped kernel when written in the traditional mathematical-convolution convention.

In practical deep-learning implementations, it is safer to think in terms of the direct dependency:

> Every input pixel receives the sum of gradient contributions from all output positions that depended on it.

---

# 24. Multi-Channel Convolution Gradients

For the general convolution:

\[
Y_{o,i,j}
=
b_o+
\sum_c\sum_u\sum_v
W_{o,c,u,v}
X_{c,i+u,j+v}
\]

the weight gradient becomes:

\[
\boxed{
\frac{\partial L}
{\partial W_{o,c,u,v}}
=
\sum_{n,i,j}
G_{n,o,i,j}
X_{n,c,i+u,j+v}
}
\]

where \(n\) indexes the batch.

The bias gradient is:

\[
\boxed{
\frac{\partial L}{\partial b_o}
=
\sum_{n,i,j}
G_{n,o,i,j}
}
\]

The input gradient for channel \(c\) accumulates contributions from **all output channels** whose filters used that input channel.

This is exactly what happens in the notebook's second convolution:

```text
32 input channels
        ↓
64 filters
        ↓
64 output channels
```

When backpropagating through that layer, each of the 32 input feature maps receives accumulated gradient contributions from all 64 output filters.

---

# 25. A Complete Numerical Backward Example

The reference reading provides manual forward examples.

This section extends that style with a compact **forward + backward** example so that every major CNN gradient can be followed numerically.

We use:

\[
X=
\begin{bmatrix}
1&2&0\\
0&1&3\\
2&1&0
\end{bmatrix}
\]

Kernel:

\[
W=
\begin{bmatrix}
1&-1\\
0&2
\end{bmatrix}
\]

Convolution bias:

\[
b=0
\]

Settings:

```text
stride = 1
padding = 0
```

Then:

```text
Conv
 ↓
ReLU
 ↓
2×2 MaxPool
 ↓
one scalar feature
 ↓
Linear → 2 logits
 ↓
Cross Entropy
```

---

# 26. Numerical Example — Convolution Forward

The output size is:

\[
\frac{3-2}{1}+1=2
\]

so the convolution output is \(2\times2\).

### Position (0,0)

Input patch:

\[
\begin{bmatrix}
1&2\\
0&1
\end{bmatrix}
\]

Therefore:

\[
Y_{0,0}
=
1(1)+2(-1)+0(0)+1(2)
\]

\[
=1-2+0+2
\]

\[
=1
\]

### Position (0,1)

Patch:

\[
\begin{bmatrix}
2&0\\
1&3
\end{bmatrix}
\]

\[
Y_{0,1}
=
2(1)+0(-1)+1(0)+3(2)
\]

\[
=2+6=8
\]

### Position (1,0)

Patch:

\[
\begin{bmatrix}
0&1\\
2&1
\end{bmatrix}
\]

\[
Y_{1,0}
=
0(1)+1(-1)+2(0)+1(2)
\]

\[
=-1+2=1
\]

### Position (1,1)

Patch:

\[
\begin{bmatrix}
1&3\\
1&0
\end{bmatrix}
\]

\[
Y_{1,1}
=
1(1)+3(-1)+1(0)+0(2)
\]

\[
=1-3=-2
\]

Therefore:

\[
Y=
\begin{bmatrix}
1&8\\
1&-2
\end{bmatrix}
\]

---

# 27. Numerical Example — ReLU and MaxPool Forward

Apply ReLU:

\[
A=
\operatorname{ReLU}(Y)
=
\begin{bmatrix}
1&8\\
1&0
\end{bmatrix}
\]

Now apply a \(2\times2\) MaxPool:

\[
P=\max(1,8,1,0)=8
\]

The winning position is:

```text
row 0, column 1
```

Flatten does nothing interesting here because there is already one value:

\[
f=[8]
\]

---

# 28. Numerical Example — Linear Classifier

Use a two-class classifier:

\[
W_{\text{fc}}
=
\begin{bmatrix}
0.2\\
-0.1
\end{bmatrix}
\]

and:

\[
b_{\text{fc}}
=
\begin{bmatrix}
0\\
0
\end{bmatrix}
\]

Then:

\[
z=W_{\text{fc}}f+b_{\text{fc}}
\]

so:

\[
z_0=0.2(8)=1.6
\]

\[
z_1=-0.1(8)=-0.8
\]

Thus:

\[
z=
\begin{bmatrix}
1.6\\
-0.8
\end{bmatrix}
\]

Assume the correct class is class 0:

\[
y=
\begin{bmatrix}
1\\
0
\end{bmatrix}
\]

---

# 29. Numerical Example — Softmax and Loss

Softmax:

\[
p_0
=
\frac{e^{1.6}}
{e^{1.6}+e^{-0.8}}
\approx0.9168
\]

\[
p_1
\approx0.0832
\]

Therefore:

\[
p
\approx
\begin{bmatrix}
0.9168\\
0.0832
\end{bmatrix}
\]

Cross Entropy:

\[
L=-\log(0.9168)
\approx0.0868
\]

Now the forward pass is complete.

---

# 30. Numerical Example — Start Backpropagation

For Softmax + Cross Entropy:

\[
\frac{\partial L}{\partial z}
=
p-y
\]

Therefore:

\[
g_z
=
\begin{bmatrix}
0.9168-1\\
0.0832-0
\end{bmatrix}
\]

\[
\boxed{
g_z
\approx
\begin{bmatrix}
-0.0832\\
0.0832
\end{bmatrix}
}
\]

---

# 31. Numerical Example — Linear Layer Gradients

Because:

\[
z=W_{\text{fc}}f+b_{\text{fc}}
\]

the weight gradient is:

\[
\frac{\partial L}{\partial W_{\text{fc}}}
=
g_zf
\]

Since:

\[
f=8
\]

we get:

\[
\frac{\partial L}{\partial W_{\text{fc}}}
\approx
\begin{bmatrix}
-0.0832(8)\\
0.0832(8)
\end{bmatrix}
\]

\[
\boxed{
\frac{\partial L}{\partial W_{\text{fc}}}
\approx
\begin{bmatrix}
-0.6654\\
0.6654
\end{bmatrix}
}
\]

Bias gradient:

\[
\boxed{
\frac{\partial L}{\partial b_{\text{fc}}}
\approx
\begin{bmatrix}
-0.0832\\
0.0832
\end{bmatrix}
}
\]

Now calculate the gradient with respect to the pooled feature:

\[
\frac{\partial L}{\partial f}
=
W_{\text{fc}}^Tg_z
\]

\[
=
0.2(-0.0832)
+
(-0.1)(0.0832)
\]

\[
=-0.01664-0.00832
\]

\[
\boxed{
\frac{\partial L}{\partial f}
\approx-0.02495
}
\]

---

# 32. Numerical Example — MaxPool Backward

The MaxPool forward step selected:

\[
A_{0,1}=8
\]

Therefore the entire incoming gradient goes back to that location:

\[
\frac{\partial L}{\partial A}
=
\begin{bmatrix}
0&-0.02495\\
0&0
\end{bmatrix}
\]

All non-winning elements receive zero.

---

# 33. Numerical Example — ReLU Backward

Recall:

\[
Y=
\begin{bmatrix}
1&8\\
1&-2
\end{bmatrix}
\]

The ReLU derivative mask is:

\[
\mathbf{1}[Y>0]
=
\begin{bmatrix}
1&1\\
1&0
\end{bmatrix}
\]

Therefore:

\[
\frac{\partial L}{\partial Y}
=
\frac{\partial L}{\partial A}
\odot
\mathbf{1}[Y>0]
\]

so:

\[
\boxed{
\frac{\partial L}{\partial Y}
=
\begin{bmatrix}
0&-0.02495\\
0&0
\end{bmatrix}
}
\]

Only one convolution output currently carries gradient.

---

# 34. Numerical Example — Kernel Gradient

The only nonzero output gradient is:

\[
G_{0,1}=-0.02495
\]

The input patch that created \(Y_{0,1}\) was:

\[
\begin{bmatrix}
2&0\\
1&3
\end{bmatrix}
\]

Therefore:

\[
\frac{\partial L}{\partial W}
=
G_{0,1}
\begin{bmatrix}
2&0\\
1&3
\end{bmatrix}
\]

\[
=
-0.02495
\begin{bmatrix}
2&0\\
1&3
\end{bmatrix}
\]

Thus:

\[
\boxed{
\frac{\partial L}{\partial W}
\approx
\begin{bmatrix}
-0.04990&0\\
-0.02495&-0.07486
\end{bmatrix}
}
\]

If multiple convolution output positions had nonzero gradients, the kernel gradient would be the **sum of all their patch contributions**.

That is how shared weights learn from the entire image.

---

# 35. Numerical Example — Convolution Bias Gradient

There is one nonzero output gradient:

\[
-0.02495
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial b}
=
-0.02495
}
\]

In a larger feature map, this would be the sum over all output spatial positions.

---

# 36. Numerical Example — Input Gradient

The nonzero gradient came from convolution output position \((0,1)\).

That output used the input patch:

```text
rows 0..1
columns 1..2
```

We multiply the kernel by:

\[
G_{0,1}=-0.02495
\]

Kernel:

\[
W=
\begin{bmatrix}
1&-1\\
0&2
\end{bmatrix}
\]

Contribution to the corresponding input window:

\[
-0.02495
\begin{bmatrix}
1&-1\\
0&2
\end{bmatrix}
=
\begin{bmatrix}
-0.02495&0.02495\\
0&-0.04990
\end{bmatrix}
\]

Place that contribution back into the correct location of the \(3\times3\) input-gradient matrix:

\[
\boxed{
\frac{\partial L}{\partial X}
\approx
\begin{bmatrix}
0&-0.02495&0.02495\\
0&0&-0.04990\\
0&0&0
\end{bmatrix}
}
\]

With several nonzero output gradients, overlapping contributions would be added together.

---

# 37. What `loss.backward()` Computes in Our Notebook

For the actual MNIST CNN:

```text
CrossEntropyLoss
      ↓
fc.weight.grad       shape = 10 × 3136
fc.bias.grad         shape = 10
      ↓
reshape gradient
      ↓
Pool2 backward       shape = 64 × 14 × 14 before pooling
      ↓
ReLU2 backward
      ↓
conv2.weight.grad    shape = 64 × 32 × 3 × 3
conv2.bias.grad      shape = 64
      ↓
Pool1 backward
      ↓
ReLU1 backward
      ↓
conv1.weight.grad    shape = 32 × 1 × 3 × 3
conv1.bias.grad      shape = 32
```

For a batch, each parameter gradient combines contributions from all examples in the batch.

Because `CrossEntropyLoss()` uses mean reduction by default, the loss gradient is averaged across the batch before the SGD update.

---

# 38. SGD Update

After:

```python
loss.backward()
```

PyTorch has stored gradients in:

```python
parameter.grad
```

Then:

```python
optimizer.step()
```

performs the SGD update.

For parameter \(\theta\):

\[
\boxed{
\theta
\leftarrow
\theta
-
\eta
\frac{\partial L}{\partial\theta}
}
\]

where:

\[
\eta
\]

is the learning rate.

The notebook uses:

\[
\eta=0.1
\]

Therefore the complete learning process is:

```text
Forward pass
      ↓
Compute logits
      ↓
Cross Entropy loss
      ↓
Backward pass
      ↓
Compute gradients for FC and convolution kernels
      ↓
SGD updates parameters
      ↓
Next batch
```

---

# 39. Why Convolution Filters Learn Useful Features

Initially, convolution filters contain random learned parameters.

They do not begin as explicit edge or curve detectors.

During training:

```text
Kernel produces feature map
      ↓
Feature map affects later layers
      ↓
Later layers affect logits
      ↓
Logits affect Cross Entropy
      ↓
Cross Entropy creates gradient
      ↓
Gradient flows back to kernel
      ↓
Kernel changes
```

If changing a kernel weight would reduce classification loss, gradient descent pushes that weight in the corresponding direction.

Over many examples, filters can become useful detectors for recurring local patterns.

The first layer often responds to relatively simple local structures, while later layers operate on earlier feature maps and can combine them into more complex representations.

---

# 40. Why Weight Sharing Matters During Backpropagation

Consider one kernel weight:

\[
W_{u,v}
\]

The same value is used at many positions.

Therefore:

\[
\frac{\partial L}{\partial W_{u,v}}
=
\sum_{i,j}
G_{i,j}X_{i+u,j+v}
\]

This means one parameter learns from many spatial locations.

Contrast this with a fully connected layer:

```text
one connection
→ one independent weight
```

In convolution:

```text
many spatial connections
→ one shared weight
```

This provides two important properties:

1. **Parameter efficiency**  
   We do not need a different kernel for every image location.

2. **Translation-related feature reuse**  
   A local pattern learned in one part of the image can also be detected elsewhere.

---

# 41. Receptive Field Intuition

A convolutional neuron only sees a local region initially.

For the baseline model:

```text
Conv 3×3
→ local neighborhood

Pool 2×2
→ combines nearby responses

Conv 3×3
→ combines information from earlier local regions

Pool 2×2
→ larger effective region
```

As layers are stacked, later features depend on increasingly large regions of the original image.

For this particular network, the theoretical receptive-field size of one final pooled spatial cell grows approximately as:

```text
Input pixel                : 1 × 1
After Conv1 (3×3)          : 3 × 3
After Pool1 (2×2, s=2)     : 4 × 4
After Conv2 (3×3)          : 8 × 8
After Pool2 (2×2, s=2)     : 10 × 10
```

Thus the CNN progressively combines local information into higher-level spatial representations.

---

# 42. What Changed from MLP?

## Reused

The following concepts are not new:

```text
MNIST
Train / validation / test
Mini-batches
ReLU
Logits
Cross Entropy
Backpropagation
SGD
Accuracy
```

The final linear layer also uses the same mathematics as the earlier models.

---

## Added

CNN introduces:

```text
Spatial tensors
Local connectivity
Kernels / filters
Parameter sharing
Feature maps
Channels
Stride
Padding
Pooling
Hierarchical spatial representation
Convolution-specific gradients
```

The main transition is:

```text
MLP
Image
 ↓
Flatten raw pixels
 ↓
Fully connected representation

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

# 43. Notebook vs Documentation

The two files intentionally have different roles.

## `notebooks/05_cnn.ipynb`

Use it to understand:

```text
How to implement the architecture
How tensors move through the model
How to train it in PyTorch
How to evaluate it
How learned feature maps look
```

It keeps code simple and lets autograd handle derivatives.

---

## `docs/05_cnn.md`

Use this document to understand:

```text
What each CNN operation calculates
Why tensor shapes change
How many parameters are learned
How Cross Entropy creates the first gradient
How gradient flows through Linear
How gradient flows through Flatten
How gradient flows through MaxPool
How gradient flows through ReLU
How kernel / bias / input gradients are calculated
Why shared filters learn from many image locations
```

Together:

```text
Notebook
= implementation intuition

Documentation
= mathematical intuition
```

---

# 44. Key Takeaways

1. CNNs process image data using spatially structured operations instead of flattening immediately.

2. A convolutional filter computes local weighted sums and reuses the same weights across image positions.

3. One filter spans all input channels and produces one output feature map.

4. The number of filters determines the number of output channels.

5. Stride, padding, and kernel size determine the spatial output dimensions.

6. ReLU introduces nonlinearity and blocks gradients where its pre-activation is non-positive.

7. MaxPool keeps the maximum value during the forward pass and routes the backward gradient only to the stored maximum location.

8. Flatten performs only a reshape; its backward pass reshapes the gradient back.

9. The final linear classifier uses the same forward and backward mathematics already encountered in MLPs.

10. For Softmax + Cross Entropy:

\[
\frac{\partial L}{\partial z}=p-y
\]

for one sample, with the appropriate averaging factor when the batch loss is averaged.

11. For convolution, the kernel gradient is formed by combining input patches with the gradients of the output feature map:

\[
\frac{\partial L}{\partial W_{u,v}}
=
\sum_{i,j}
G_{i,j}X_{i+u,j+v}
\]

12. Shared convolution weights accumulate gradient contributions from every spatial location where they were used.

13. The input gradient accumulates contributions from every convolution output that depended on each input value.

14. `loss.backward()` in PyTorch automatically performs this complete chain of derivatives.

15. Our baseline CNN contains:

\[
\boxed{50,186}
\]

trainable parameters.

---

# Reference

Primary forward-pass reference:

**AI VIET NAM – AI Course 2025**  
**_CNNs: Step-by-Step Examples_**  
Nguyễn Phúc Thịnh and Đinh Quang Vinh

Topics used from the reference include:

- motivation for CNNs,
- loss of explicit spatial structure after flattening,
- convolution,
- stride,
- padding,
- Max / Average Pooling,
- flattening,
- image channels,
- and manual forward-pass calculations.

The backward-pass derivations and the complete numerical backward example in this document are additional companion material written to explain the operations that PyTorch autograd performs automatically in `notebooks/05_cnn.ipynb`.
