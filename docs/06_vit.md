# Vision Transformer on MNIST

This is the revision companion to `notebooks/06_vit.ipynb` and `src/06_vit.py`. It explains the new ideas behind the implementation; the notebook contains the code walkthrough and visualizations.

## 1. What changes from CNN?

A CNN builds spatial feature maps using local filters shared across image positions. Our ViT divides the image into patches, represents each patch as a vector, and lets those vectors exchange information through attention.

A **token** is one vector in this sequence. An **embedding** is its learned numerical representation. Neither term requires text: our tokens represent image patches.

The classifier uses only a Transformer **encoder**, which transforms an input sequence into contextual representations. A decoder for generating an output sequence is unnecessary here.

| Operation | Output shape |
|---|---|
| MNIST images | `(B, 1, 28, 28)` |
| Extract sixteen 7 × 7 patches | `(B, 16, 49)` |
| Project each patch into 64 features | `(B, 16, 64)` |
| Add CLS token and positional embeddings | `(B, 17, 64)` |
| Two independent encoder blocks | `(B, 17, 64)` |
| Final LayerNorm; select CLS | `(B, 64)` |
| Linear classification head | `(B, 10)` |

Here, `B` is the batch size. The feature dimension stays 64 inside the encoder so residual additions have matching shapes.

## 2. Patches and their embeddings

For image height $H$, width $W$, channels $C$, and square patch size $P$:

$$
N = (H/P)(W/P), \qquad D_{\text{patch}} = CP^2.
$$

Our image gives $N=16$ patches, each with 49 pixel values. Extraction only rearranges values: $16\times49=784$. It has no parameters and discards no pixels.

One shared linear layer then maps every patch $x_i$ to an embedding:

$$
e_i=x_iW_E+b_E,
\qquad W_E\in\mathbb{R}^{49\times64}.
$$

Each of the 64 features is a learned weighted combination of the patch pixels. These features are not assigned meanings such as “curve” beforehand. The classification loss determines what becomes useful.

`nn.Linear(49, 64)` operates on the last dimension. PyTorch stores its weight as `(64, 49)` and computes `x @ weight.T + bias`.

Increasing 49 values to 64 features does not create new image information; it provides a learned representation for later operations.

## 3. CLS and positional embeddings

The **classification token (CLS)** is a trainable vector prepended to the patch sequence. Every image begins with the same CLS vector. Attention makes its later representation depend on the image; the final head reads this representation.

CLS is not a class label, an image patch, or an average computed in advance.

**Positional embeddings** are learned vectors added to the sequence positions:

$$
Z_0=[x_{\text{CLS}};e_0;\ldots;e_{15}]+E_{\text{pos}},
\qquad E_{\text{pos}}\in\mathbb{R}^{1\times17\times64}.
$$

The semicolons mean concatenation along the sequence. Addition combines content with position without increasing feature width. Positions follow the notebook's left-to-right, top-to-bottom patch order.

Without positional information, rearranging patch tokens would rearrange the encoder's patch outputs but leave its CLS output unchanged. The model would lack an explicit way to distinguish patch arrangements.

CLS and positional vectors are initialized with small normal random values, mean 0 and standard deviation 0.02. They are optimized through backpropagation like other weights. This implementation follows the basic ViT token design [1].

## 4. Self-attention: how tokens exchange information

For normalized input tokens $U$, three learned linear projections produce:

$$
Q=UW_Q+b_Q,\qquad K=UW_K+b_K,\qquad V=UW_V+b_V.
$$

| Vector | Intuition |
|---|---|
| Query $q_i$ | What information token $i$ seeks |
| Key $k_j$ | What token $j$ can match against |
| Value $v_j$ | Information contributed by token $j$ |

These are learned features, not literal questions or fixed labels. Although Q, K, and V come from the same sequence, their projections differ.

For one head:

$$
S=\frac{QK^T}{\sqrt{d_h}},\qquad
A=\operatorname{softmax}_{\text{row}}(S),\qquad
O=AV.
$$

Entry $A_{ij}$ weights source token $j$ when updating receiver token $i$:

$$
o_i=\sum_j A_{ij}v_j.
$$

For example, weights `[0.7, 0.2, 0.1]` produce $0.7v_0+0.2v_1+0.1v_2$. Each row sums to one before any attention dropout.

**Why divide by $\sqrt{d_h}$?** Under a simple independent, unit-variance model, a dot product's variance grows with $d_h$. Scaling controls its magnitude so Softmax is less likely to become excessively peaked with tiny gradients [2]. Here $d_h=16$, so we divide by 4.

Attention weights depend on the current image and layer. They are activations computed from learned projections, not a fixed trainable 17 × 17 matrix. A heatmap is useful for inspection but does not fully explain a prediction.

## 5. Multiple heads, masks, and sequence length

Four heads give four learned ways to combine token information. Each head sees all 17 tokens through 16-dimensional Q, K, and V vectors. Heads do not receive separate groups of patches.

$$
\operatorname{MSA}(U)
=\operatorname{Concat}(O_1,O_2,O_3,O_4)W_O+b_O.
$$

| Intermediate | Shape |
|---|---|
| Q, K, V after splitting heads | `(B, 4, 17, 16)` |
| Attention scores/weights | `(B, 4, 17, 17)` |
| Concatenated head outputs | `(B, 17, 64)` |
| Output projection | `(B, 17, 64)` |

`nn.MultiheadAttention` performs these projections internally. `batch_first=True` specifies `(batch, sequence, features)`. `need_weights=False` skips returning the attention weights; it does not disable attention. The notebook requests per-head weights for inspection [3].

An **attention mask** excludes selected token interactions. We need no causal mask: the complete image is available, and every token may attend to every other token, including itself. All images have equal token counts, so no padding mask is needed either.

Attention computes $T^2$ token-pair scores per head. Smaller patches increase $T$, potentially providing finer spatial units at greater computational cost.

## 6. Layer Normalization

`LayerNorm(64)` normalizes the 64 features of each token separately [4]. For one token $x$ with $D=64$:

$$
\mu=\frac1D\sum_jx_j,\qquad
\sigma^2=\frac1D\sum_j(x_j-\mu)^2,
$$

$$
\operatorname{LN}(x)_j
=\gamma_j\frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_j.
$$

- $\epsilon=10^{-5}$ stabilizes division when variance is small.
- $\gamma$ and $\beta$ are learned scale and shift vectors, initialized to 1 and 0.
- There are $64+64=128$ parameters per LayerNorm.

Centering and scaling the features helps control the inputs to attention and the feed-forward network. Learned scale and shift preserve flexibility; the final output need not have zero mean and unit variance.

Example: ignoring epsilon, `[1, 2, 3]` becomes approximately `[-1.225, 0, 1.225]` before learned scaling and shifting.

This is different from pixel scaling by 255. Pixel scaling prepares the input; LayerNorm acts on intermediate token features. Unlike BatchNorm, it does not estimate statistics across the batch or maintain running averages. It uses the current token's statistics during both training and evaluation.

## 7. Residual connections and pre-normalization

A residual connection adds the input to a learned transformation:

$$
y=x+F(x).
$$

The branch learns a correction to the existing representation. The derivative contains a direct identity path:

$$
\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}.
$$

This helps information and gradients pass through stacked blocks, though it does not guarantee perfect optimization.

Our block applies LayerNorm **before** each transformation, hence **pre-norm**:

$$
Z'=Z+\operatorname{MSA}(\operatorname{LN}_1(Z)),
$$

$$
Z_{\text{out}}=Z'+\operatorname{FFN}(\operatorname{LN}_2(Z')).
$$

The unchanged branch carries the unnormalized input. The two LayerNorm modules have separate parameters. A final LayerNorm is applied after both encoder blocks, as in the ViT design [1].

## 8. Feed-forward network and GELU

The **feed-forward network (FFN)** is a small MLP applied independently to each token, with shared weights across positions:

$$
\operatorname{FFN}(u)
=\operatorname{GELU}(uW_1+b_1)W_2+b_2.
$$

Its dimensions are $64\rightarrow128\rightarrow64$. Attention combines different tokens; the FFN mixes features within each token. Returning to 64 dimensions permits the residual addition.

**GELU**, Gaussian Error Linear Unit, is the nonlinear activation [5]:

$$
\operatorname{GELU}(x)=x\Phi(x),
$$

where $\Phi(x)$ is the standard normal cumulative distribution function: the probability that a standard normal variable is at most $x$.

ReLU sets all negative inputs to zero. GELU smoothly scales inputs: large positive values pass almost unchanged, while negative values are suppressed smoothly.

| Input | ReLU | GELU, approximately |
|---:|---:|---:|
| −1 | 0 | −0.159 |
| 0 | 0 | 0 |
| 1 | 1 | 0.841 |

GELU is deterministic; it does not randomly keep or discard values. Its derivative is $\Phi(x)+x\phi(x)$, where $\phi$ is the standard normal density. Autograd calculates the backward pass.

## 9. Classification and backpropagation

After the final LayerNorm, select CLS at index 0 and compute:

$$
z=h_{\text{CLS}}W_{\text{head}}+b_{\text{head}}.
$$

The output is ten logits. Cross Entropy and its familiar batch gradient remain:

$$
\frac{\partial L}{\partial z_{ik}}
=\frac{p_{ik}-\mathbf{1}[y_i=k]}{B}.
$$

`loss.backward()` propagates this signal through the head, normalization, encoder blocks, token preparation, and patch projection. Attention lets the CLS loss influence patch representations even though the head reads only CLS.

To connect attention with ordinary matrix backpropagation, for $O=AV$ and incoming gradient $G=\partial L/\partial O$:

$$
\frac{\partial L}{\partial V}=A^TG,
\qquad
\frac{\partial L}{\partial A}=GV^T.
$$

The second path continues through row-wise Softmax and $QK^T$ into Q and K. All three projection families therefore learn. Shared parameters accumulate gradients from the positions and images using them; patch extraction simply routes gradients back to their pixel positions.

## 10. Adam: the new optimizer

SGD uses the current gradient directly. **Adam** also tracks an exponentially weighted gradient average $m_t$ and squared-gradient average $v_t$ [6]. For gradient $g_t$, starting with $m_0=v_0=0$:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

The first tracks direction; the second tracks gradient magnitude. Zero initialization biases early averages toward zero, so Adam corrects them:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
$$

The elementwise update is:

$$
\theta_t=\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Our settings are $\eta=0.001$, $\beta_1=0.9$, $\beta_2=0.999$, and $\epsilon=10^{-8}$. Here $t$ counts optimizer steps, not epochs.

Adam adapts update scales per parameter; the learning rate still matters. With a first gradient of 0.2, the corrected estimates are 0.2 and 0.04, giving an update of approximately 0.001. This is an illustrative first step, not a constant future step size.

`optimizer.zero_grad()` clears accumulated parameter gradients; it does not erase Adam's moving averages. `optimizer.step()` updates parameters and those averages. We use zero weight decay, so this baseline adds no weight-decay penalty.

## 11. Code details worth understanding

| Code | Meaning in this implementation |
|---|---|
| `unfold(..., size=P, step=P)` | Extract non-overlapping patch windows. |
| `permute` then `reshape` | Order patch positions and flatten each patch's pixels. |
| `nn.Parameter` | Register CLS/position tensors as trainable model parameters. |
| `expand(B, -1, -1)` | Reuse CLS across the batch without creating independent learned copies. |
| Broadcasting `(1, 17, 64)` | Add the same positional embeddings to each image. |
| `nn.ModuleList` | Register independently constructed encoder blocks so their parameters are optimized and moved with the model. |
| `.to(device)` | Put model/data on CPU or GPU; labels must be on the same device as logits. |
| `torch.cuda.synchronize()` | Wait for queued GPU work when measuring elapsed time. |

**Dropout** randomly suppresses selected activations during training for regularization. Attention dropout is set to 0, and our FFN has no dropout layer. No scheduler, warmup, gradient clipping, or pretraining is used.

`model.eval()` switches mode-sensitive layers to evaluation behavior; `torch.no_grad()` disables gradient tracking. They serve different purposes. LayerNorm itself uses the same normalization rule in both modes.

## 12. Parameters, results, and interpretation

| Component | Calculation | Parameters |
|---|---|---:|
| Patch projection | $49\times64+64$ | 3,200 |
| CLS + positions | $64+17\times64$ | 1,152 |
| Attention per block | $3(64^2+64)+(64^2+64)$ | 16,640 |
| FFN per block | $(64\times128+128)+(128\times64+64)$ | 16,576 |
| Two LayerNorms per block | $2(64+64)$ | 256 |
| Two complete blocks | $2\times33,472$ | 66,944 |
| Final LayerNorm | $64+64$ | 128 |
| Classification head | $64\times10+10$ | 650 |
| **Model total** | $3,200+1,152+66,944+128+650$ | **72,074** |

Recorded notebook results, reported on 2026-10-03:

| Model | Parameters | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| NumPy Softmax, selected | 7,850 | 92.05% | 92.54% |
| MLP, 128 hidden units | 101,770 | 97.11% | 97.44% |
| CNN baseline | 50,186 | 98.72% | 98.95% |
| ViT baseline | 72,074 | 97.24% | 97.45% |

These are recorded notebook runs, not new results from executing `src/06_vit.py`. ViT used 50,000 training images, batch size 32, ten epochs, and Adam at 0.001.

CNN exceeded ViT by 1.50 percentage points on the test set. A plausible explanation is **inductive bias**: CNN builds in local connectivity and spatial weight sharing, useful assumptions for digit strokes. **Pretraining** means learning weights on another dataset before adapting them; our ViT started from random weights. The original ViT paper demonstrated benefits from large-scale pretraining [1].

Our run does not prove that data size caused the gap. Patch size, model dimensions, training duration, and the optimizer also matter. MLP/CNN used SGD; NumPy Softmax also used a different split procedure and selected settings. ViT's near-equality with MLP does not imply equivalent representations.

Training metrics were collected during updates; validation metrics use end-of-epoch parameters. Final evaluation measures a fixed model. Runtime includes per-epoch validation, so it is not directly comparable with earlier training-only timings. A seed controls selected randomness but does not guarantee identical results across hardware and software environments.

This stage completes the basic architectural progression. Systematic experiments remain focused on Softmax and MLP; advanced CNN/ViT work belongs in the later computer-vision project.

## References

1. Dosovitskiy et al., [An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929). Architecture and data-scale context; our MNIST model is a small adaptation.
2. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Scaled dot-product attention and multiple heads.
3. PyTorch, [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html). API, projections, shapes, and returned attention weights.
4. PyTorch, [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html). Normalization dimensions, parameters, and epsilon.
5. PyTorch, [GELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html). Activation definition.
6. PyTorch, [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html). Update equations and default coefficients.
