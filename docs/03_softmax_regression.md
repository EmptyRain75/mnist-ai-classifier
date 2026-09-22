# Softmax Regression from Scratch

This document contains the deeper mathematical foundations behind the Softmax Regression implementation developed for MNIST.

The notebook `notebooks/02_softmax_regression.ipynb` is the main learning and experiment document. It focuses on implementation, visualizations, experiments, observations, and model selection.

This document focuses on the theory and derivations that would make the notebook too dense.

It assumes familiarity with the Logistic Regression foundations developed in `docs/02_logistic_regression.md`, including:

- linear models
- Gradient Descent
- Binary Cross Entropy
- basic backpropagation
- analytical and numerical gradients

The main purpose here is to explain what changes when binary Logistic Regression is extended to multiclass classification.

---

# 1. From Logistic Regression to Softmax Regression

Softmax regression is the direct generalization of binary logistic regression to multi-class problems, which is why it is often called multinomial logistic regression.

Binary Logistic Regression produces one logit:

$$ z = x^T w + b $$

and converts it into one probability using Sigmoid.

For a multiclass problem with $C$ classes, one scalar score is no longer sufficient.

The model must produce one score for every class:

$$ z = [z_0,z_1,\dots,z_{C-1}] $$

For MNIST:

$$ C = 10 $$

so each input image produces ten logits.

The scalar weight vector becomes a weight matrix:

$$ W \in \mathbb{R}^{D \times C} $$

and the scalar bias becomes a vector:

$$ b \in \mathbb{R}^{C} $$

For MNIST:

$$ D = 28 \times 28 = 784 $$

therefore:

$$ W \in \mathbb{R}^{784 \times 10} $$

The multiclass linear transformation is:

$$ Z = XW + b $$

---

# 2. Interpretation of the Weight Matrix

The weight matrix can be written as:

$$ W = [w_0\;w_1\;\dots\;w_{C-1}] $$

where each column:

$$ w_k \in \mathbb{R}^{D} $$

is associated with one class.

For one input vector $x$, the score for class $k$ is:

$$ z_k = x^T w_k + b_k $$

For MNIST:

$$ z_0 = x^T w_0 + b_0 $$

$$ z_1 = x^T w_1 + b_1 $$

$$ \dots $$

$$ z_9 = x^T w_9 + b_9 $$

The vector of logits is therefore:

$$ z = [z_0,z_1,\dots,z_9] $$

Each class has its own linear scoring function.

Softmax Regression can therefore be understood as multiple class-specific linear scorers trained jointly.

---

# 3. Batch Dimensions

For $N$ samples:

$$ X \in \mathbb{R}^{N \times D} $$

$$ W \in \mathbb{R}^{D \times C} $$

$$ b \in \mathbb{R}^{C} $$

The matrix product is:

$$ XW \in \mathbb{R}^{N \times C} $$

and therefore:

$$ Z = XW+b \in \mathbb{R}^{N \times C} $$

Each row of $Z$ corresponds to one sample.

Each column corresponds to one class.

For MNIST:

$$ X \in \mathbb{R}^{N \times 784} $$

$$ W \in \mathbb{R}^{784 \times 10} $$

$$ Z \in \mathbb{R}^{N \times 10} $$

---

# 4. The Softmax Function

The logits are arbitrary real-valued scores.

They are not probabilities because they:

- can be negative
- can exceed 1
- do not necessarily sum to 1

Softmax converts a vector of logits into a probability distribution.

For class $k$:

$$ p_k = \frac{e^{z_k}}{\sum_{j=0}^{C-1} e^{z_j}} $$

The resulting probabilities satisfy:

$$ 0 < p_k < 1 $$

and:

$$ \sum_{k=0}^{C-1} p_k = 1 $$

For MNIST:

$$ P(y=k\mid x)=p_k $$

for:

$$ k\in\{0,1,\dots,9\} $$

---

# 5. Why Softmax Uses Exponentials

The exponential function has two useful properties.

First:

$$ e^z > 0 $$

for every real number $z$.

Therefore every class receives a positive unnormalized score.

Second, the exponential preserves ordering:

$$ z_a > z_b \Rightarrow e^{z_a} > e^{z_b} $$

Therefore classes with larger logits receive larger probability mass.

The normalization:

$$ \sum_j e^{z_j} $$

forces the resulting values to sum to 1.

Softmax is therefore not simply applying a function independently to each logit.

Every output probability depends on all logits through the shared denominator.

---

# 6. Softmax Shift Invariance

Directly evaluating exponentials can cause numerical overflow.

For example:

$$ e^{1000} $$

is too large for ordinary floating-point representation.

A key property of Softmax is that subtracting the same constant from every logit does not change the probabilities.

Let $c$ be any constant.

Then:

$$ \text{softmax}(z_k-c)=\frac{e^{z_k-c}}{\sum_j e^{z_j-c}} $$

Using:

$$ e^{z_k-c}=\frac{e^{z_k}}{e^c} $$

we get:

$$ \text{softmax}(z_k-c)=\frac{e^{z_k}/e^c}{\sum_j e^{z_j}/e^c} $$

The common factor $1/e^c$ cancels:

$$ \text{softmax}(z_k-c)=\frac{e^{z_k}}{\sum_j e^{z_j}} $$

Therefore:

$$ \boxed{\text{softmax}(z)=\text{softmax}(z-c)} $$

The standard stable implementation chooses:

$$ c=\max(z) $$

so:

$$ z_{\text{shifted}}=z-\max(z) $$

The largest shifted logit becomes zero, and therefore its exponential becomes:

$$ e^0=1 $$

All other exponentials are at most 1.

This greatly reduces overflow risk.

---

# 7. Stable Batch Softmax

For a batch:

$$ Z \in \mathbb{R}^{N \times C} $$

Softmax must be applied independently to every row.

For sample $i$ and class $k$:

$$ p_{ik}=\frac{e^{z_{ik}}}{\sum_j e^{z_{ij}}} $$

The implementation subtracts the maximum separately for each sample:

```python
Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
```

Then:

```python
exp_Z = np.exp(Z_shifted)
P = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
```

The output shape remains:

$$ P \in \mathbb{R}^{N \times C} $$

and each row satisfies:

$$ \sum_k p_{ik}=1 $$

---

# 8. One-Hot Encoding

MNIST labels are integer class indices:

$$ y\in\{0,1,\dots,9\} $$

For multiclass mathematics, it is useful to represent the target as a vector.

If the correct class is 3:

$$ y=3 $$

the one-hot representation is:

$$ Y=[0,0,0,1,0,0,0,0,0,0] $$

For one sample:

$$ Y\in\mathbb{R}^{C} $$

For a batch:

$$ Y\in\mathbb{R}^{N\times C} $$

Only the correct class has value 1.

Every other class has value 0.

This representation becomes especially useful in the gradient derivation.

---

# 9. Multiclass Cross Entropy

For one sample:

$$ L=-\sum_{k=0}^{C-1}y_k\log(p_k) $$

Because $Y$ is one-hot encoded, only the correct class contributes.

If the correct class is $c$:

$$ y_c=1 $$

and:

$$ y_k=0 \quad \text{for } k\neq c $$

Therefore:

$$ L=-\log(p_c) $$

This means Cross Entropy measures the negative logarithm of the probability assigned to the correct class.

A confident correct prediction produces a small loss.

A confident incorrect prediction produces a large loss.

---

# 10. Cross Entropy for a Batch

For $N$ samples:

$$ L=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=0}^{C-1}y_{ik}\log(p_{ik}) $$

Because each target is one-hot encoded:

$$ L=-\frac{1}{N}\sum_{i=1}^{N}\log(p_{i,y_i}) $$

The implementation can therefore use integer labels directly:

```python
correct_class_probs = P[np.arange(n_samples), y]
```

followed by:

```python
loss = -np.mean(np.log(correct_class_probs))
```

This avoids constructing a one-hot matrix solely for the loss calculation.

---

# 11. Numerical Stability in Cross Entropy

Cross Entropy contains:

$$ \log(p) $$

but:

$$ \log(0) $$

is undefined.

Softmax theoretically produces strictly positive probabilities, but floating-point arithmetic can produce values extremely close to zero.

Therefore the implementation clips probabilities:

```python
correct_class_probs = np.clip(
    correct_class_probs,
    1e-12,
    1.0
)
```

This is a numerical protection mechanism.

It does not change the mathematical definition of Cross Entropy.

---

# 12. Initial Loss of a Uniform Classifier

When all weights and biases are initialized to zero:

$$ Z=0 $$

for every class.

Therefore:

$$ e^{z_k}=1 $$

and Softmax gives:

$$ p_k=\frac{1}{C} $$

For MNIST:

$$ p_k=\frac{1}{10}=0.1 $$

The initial Cross Entropy loss is:

$$ L=-\log(0.1) $$

so:

$$ L=\log(10)\approx2.3026 $$

More generally, a uniform classifier over $C$ classes has:

$$ L=\log(C) $$

This provides a useful implementation sanity check.

---

# 13. Derivative of Softmax

Unlike Sigmoid, each Softmax output depends on every logit.

For:

$$ p_k=\frac{e^{z_k}}{\sum_j e^{z_j}} $$

the derivative depends on whether we differentiate with respect to the same logit or a different logit.

For $k=j$:

$$ \frac{\partial p_k}{\partial z_k}=p_k(1-p_k) $$

For $k\neq j$:

$$ \frac{\partial p_k}{\partial z_j}=-p_kp_j $$

These two cases can be written compactly as:

$$ \frac{\partial p_k}{\partial z_j}=p_k(\delta_{kj}-p_j) $$

where $\delta_{kj}$ is the Kronecker delta:

$$ \delta_{kj}=1 \text{ if } k=j $$

$$ \delta_{kj}=0 \text{ if } k\neq j $$

The full derivative is therefore a Jacobian matrix rather than a single scalar derivative.

---

# 14. Softmax Jacobian

For a probability vector:

$$ p=[p_1,p_2,\dots,p_C] $$

the Softmax Jacobian is:

$$ J=\frac{\partial p}{\partial z} $$

and can be written as:

$$ J=\mathrm{diag}(p)-pp^T $$

For example, with three classes:

$$ J=\begin{bmatrix} p_1(1-p_1) & -p_1p_2 & -p_1p_3 \\ -p_2p_1 & p_2(1-p_2) & -p_2p_3 \\ -p_3p_1 & -p_3p_2 & p_3(1-p_3) \end{bmatrix} $$

This structure reflects the fact that changing one logit changes every Softmax probability.

Fortunately, when Softmax is combined with Cross Entropy, the gradient simplifies dramatically.

---

# 15. Deriving Softmax + Cross Entropy

For one sample:

$$ L=-\sum_k y_k\log(p_k) $$

We want:

$$ \frac{\partial L}{\partial z_j} $$

Using the chain rule:

$$ \frac{\partial L}{\partial z_j}=\sum_k\frac{\partial L}{\partial p_k}\frac{\partial p_k}{\partial z_j} $$

First:

$$ \frac{\partial L}{\partial p_k}=-\frac{y_k}{p_k} $$

and:

$$ \frac{\partial p_k}{\partial z_j}=p_k(\delta_{kj}-p_j) $$

Substitute:

$$ \frac{\partial L}{\partial z_j}=\sum_k\left(-\frac{y_k}{p_k}\right)p_k(\delta_{kj}-p_j) $$

Cancel $p_k$:

$$ \frac{\partial L}{\partial z_j}=-\sum_k y_k(\delta_{kj}-p_j) $$

Distribute:

$$ \frac{\partial L}{\partial z_j}=-\sum_k y_k\delta_{kj}+\sum_k y_kp_j $$

The first term selects $y_j$:

$$ \sum_k y_k\delta_{kj}=y_j $$

Because one-hot targets sum to 1:

$$ \sum_k y_k=1 $$

therefore:

$$ \sum_k y_kp_j=p_j $$

So:

$$ \frac{\partial L}{\partial z_j}=-y_j+p_j $$

and finally:

$$ \boxed{\frac{\partial L}{\partial z_j}=p_j-y_j} $$

For the full output vector:

$$ \boxed{\frac{\partial L}{\partial z}=p-y} $$

For a batch:

$$ \boxed{dZ=P-Y} $$

This is the multiclass equivalent of the Logistic Regression result.

---

# 16. Why the Gradient $P-Y$ Makes Sense

Suppose the target class is class 2:

$$ Y=[0,0,1,0] $$

and the model predicts:

$$ P=[0.10,0.20,0.60,0.10] $$

Then:

$$ P-Y=[0.10,0.20,-0.40,0.10] $$

For incorrect classes, the gradient values are positive.

Gradient Descent subtracts the gradient, reducing those logits.

For the correct class:

$$ 0.60-1=-0.40 $$

the gradient is negative.

Subtracting a negative gradient increases the corresponding logit.

Therefore training pushes probability mass toward the correct class.

---

# 17. Gradient with Respect to the Weight Matrix

The linear layer is:

$$ Z=XW+b $$

For one sample:

$$ z=xW+b $$

The upstream gradient is:

$$ dZ=P-Y $$

For a batch of $N$ samples:

$$ \boxed{dW=\frac{1}{N}X^T(P-Y)} $$

Check the dimensions:

$$ X^T\in\mathbb{R}^{D\times N} $$

$$ P-Y\in\mathbb{R}^{N\times C} $$

therefore:

$$ dW\in\mathbb{R}^{D\times C} $$

which matches:

$$ W\in\mathbb{R}^{D\times C} $$

For MNIST:

$$ dW\in\mathbb{R}^{784\times10} $$

---

# 18. Gradient with Respect to the Bias Vector

The bias vector is added to every sample.

Therefore the batch bias gradient is:

$$ \boxed{db=\frac{1}{N}\sum_{i=1}^{N}(P_i-Y_i)} $$

Equivalently:

$$ \boxed{db=\mathrm{mean}(P-Y,\text{ axis}=0)} $$

The result has shape:

$$ db\in\mathbb{R}^{C} $$

For MNIST:

$$ db\in\mathbb{R}^{10} $$

---

# 19. Complete Backward Pass

The complete forward pass is:

$$ Z=XW+b $$

$$ P=\mathrm{softmax}(Z) $$

$$ L=\mathrm{CrossEntropy}(Y,P) $$

The backward pass is:

$$ dZ=P-Y $$

$$ dW=\frac{1}{N}X^TdZ $$

$$ db=\frac{1}{N}\sum_i dZ_i $$

This gives all gradients required by Gradient Descent.

---

# 20. Gradient Descent

The parameter update rules are unchanged from Logistic Regression:

$$ W\leftarrow W-\eta dW $$

$$ b\leftarrow b-\eta db $$

The main difference is that the parameters and gradients are now matrices and vectors representing multiple classes.

The optimization principle itself is unchanged.

---

# 21. Numerical Gradient Checking

Analytical gradients are verified using the central finite-difference approximation.

For one parameter $\theta$:

$$ \frac{\partial L}{\partial\theta}\approx\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon} $$

The numerical gradient is computed independently of the analytical gradient formula.

If:

$$ \text{analytical gradient}\approx\text{numerical gradient} $$

then the implementation is likely correct.

For MNIST, it is important to choose a weight connected to a nonzero input pixel.

A weight attached to a pixel that is zero for every checked sample naturally has a zero gradient.

Therefore, a meaningful gradient check can select a parameter with a clearly nonzero analytical gradient.

---

# 22. Full-Batch Gradient Descent

Full-batch Gradient Descent uses the entire training dataset for each parameter update.

For one iteration:

$$ X_{\text{train}}\rightarrow P\rightarrow L\rightarrow dW,db\rightarrow\text{update} $$

With $N=50000$ training samples, every update processes all 50000 samples.

Advantages:

- deterministic updates
- simple implementation
- smooth loss trajectory

Disadvantages:

- expensive updates
- only one update per complete pass through the dataset
- inefficient for larger datasets

---

# 23. Mini-Batch Gradient Descent

Mini-batch Gradient Descent divides the dataset into smaller subsets.

For batch size $B$:

$$ \text{updates per epoch}\approx\frac{N}{B} $$

For example, with 50000 samples:

$$ B=32\Rightarrow\text{about }1563\text{ updates per epoch} $$

while:

$$ B=2048\Rightarrow\text{about }25\text{ updates per epoch} $$

Mini-batch training therefore changes both:

- update frequency
- computational structure

It can provide a practical balance between full-batch and single-sample SGD.

---

# 24. SGD vs Mini-Batch Gradient Descent

The terminology is:

**Stochastic Gradient Descent**

$$ B=1 $$

One sample is used per parameter update.

**Mini-Batch Gradient Descent**

$$ 1<B<N $$

A subset of samples is used per update.

**Full-Batch Gradient Descent**

$$ B=N $$

The complete dataset is used per update.

In modern machine learning, the term "SGD" is sometimes used informally for mini-batch optimization, but mathematically they are distinct cases.

---

# 25. Learning Rate and Batch Size Interaction

The best learning rate for full-batch training does not necessarily remain appropriate for mini-batch training.

Full-batch gradients are averages over the complete dataset.

Mini-batch gradients are estimates based on subsets of the data.

Their noise and update frequency differ.

Therefore:

$$ \text{learning rate} $$

and:

$$ \text{batch size} $$

should not always be optimized independently and then combined blindly.

This explains why a learning rate that works well in the full-batch experiment may behave poorly with small mini-batches.

---

# 26. Accuracy and Cross Entropy Measure Different Things

Accuracy depends only on the largest predicted probability:

$$ \hat y=\arg\max_k p_k $$

Cross Entropy depends on the probability assigned to the correct class:

$$ L=-\log(p_{\text{correct}}) $$

Therefore two models can have similar accuracy but different Cross Entropy losses.

For example, both models may classify a sample correctly:

$$ P_A=[0.01,0.90,0.09] $$

$$ P_B=[0.20,0.41,0.39] $$

Both predict class 1.

However, model $A$ is much more confident in the correct class and therefore receives a smaller Cross Entropy loss.

Cross Entropy also strongly penalizes highly confident incorrect predictions.

Therefore validation accuracy and loss should be interpreted together.

---

# 27. Why Very Large Learning Rates Become Unstable

Gradient Descent performs:

$$ W_{\text{new}}=W-\eta dW $$

If $\eta$ becomes too large, the update step can move far beyond a useful low-loss region.

Instead of approaching a minimum gradually, parameters may repeatedly overshoot.

This can produce:

- rapidly changing logits
- poor probability distributions
- high Cross Entropy
- degraded accuracy
- sensitivity to very small numerical differences

This explains why extremely large learning rates can produce unstable or irregular results.

---

# 28. Reproducibility and Floating-Point Computation

With fixed data, initialization, and deterministic code, stable configurations should produce essentially identical model results.

Runtime is different.

Wall-clock time depends on:

- current machine load
- CPU scheduling
- numerical library implementation
- cache behavior
- background processes

Therefore runtime measurements can vary between executions.

Extremely unstable optimization can also amplify tiny floating-point differences.

Normally these differences are negligible.

With excessively large parameter updates, however, tiny numerical differences can eventually lead to noticeably different trajectories.

---

# 29. Why Softmax Regression Is Still Linear

Softmax is nonlinear, but the class scores are produced by:

$$ Z=XW+b $$

The model does not learn an intermediate nonlinear representation of the input.

For two classes $a$ and $b$, the decision boundary occurs where:

$$ z_a=z_b $$

Therefore:

$$ x^Tw_a+b_a=x^Tw_b+b_b $$

Rearranging:

$$ x^T(w_a-w_b)+(b_a-b_b)=0 $$

This is a linear decision boundary in the original input space.

Therefore Softmax Regression remains a linear classifier.

---

# 30. Limitation on MNIST

Each MNIST image is flattened into:

$$ x\in\mathbb{R}^{784} $$

Softmax Regression directly maps those pixel values to class logits.

It does not explicitly model:

- neighboring pixel relationships
- local edges
- shapes
- translation patterns
- nonlinear feature combinations

Despite this limitation, it provides a strong baseline and a clear foundation for multiclass classification.

---

# 31. Transition to an MLP

Softmax Regression performs:

$$ X\rightarrow XW+b\rightarrow\mathrm{Softmax} $$

An MLP introduces hidden layers:

$$ X\rightarrow\text{Linear}\rightarrow\text{Activation}\rightarrow\text{Hidden Representation}\rightarrow\text{Linear}\rightarrow\mathrm{Softmax} $$

The important new element is the nonlinear hidden representation.

Concepts that remain reusable include:

- Softmax output
- Cross Entropy
- Gradient Descent
- mini-batch training
- train / validation / test separation
- evaluation
- numerical gradient checking

The main new challenge is backpropagating through multiple layers.

---

# 32. Key Equations

### Multiclass linear model

$$ \boxed{Z=XW+b} $$

### Softmax

$$ \boxed{p_k=\frac{e^{z_k}}{\sum_j e^{z_j}}} $$

### Stable Softmax shift

$$ \boxed{Z_{\text{shifted}}=Z-\max(Z)} $$

### Multiclass Cross Entropy

$$ \boxed{L=-\frac{1}{N}\sum_i\sum_k y_{ik}\log(p_{ik})} $$

### Softmax derivative

$$ \boxed{\frac{\partial p_k}{\partial z_j}=p_k(\delta_{kj}-p_j)} $$

### Softmax + Cross Entropy gradient

$$ \boxed{dZ=P-Y} $$

### Weight gradient

$$ \boxed{dW=\frac{1}{N}X^T(P-Y)} $$

### Bias gradient

$$ \boxed{db=\frac{1}{N}\sum_i(P_i-Y_i)} $$

### Gradient Descent

$$ \boxed{W\leftarrow W-\eta dW} $$

$$ \boxed{b\leftarrow b-\eta db} $$

### Prediction

$$ \boxed{\hat y=\arg\max_k p_k} $$
