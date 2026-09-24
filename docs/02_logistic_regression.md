# Logistic Regression from Scratch

This document contains the deeper mathematical foundations behind the Logistic Regression implementation.

The notebook `notebooks/01_logistic_regression.ipynb` is intended to be the readable implementation and experiment walkthrough. This document keeps the longer mathematical derivations here so that the notebook can remain focused on implementation, observations, and experiments.

---

## 1. Problem Definition

Logistic Regression is a supervised learning algorithm for classification.

For binary classification, each target belongs to one of two classes:

$$ y \in \{0,1\} $$

Given an input vector $x$, the model estimates:

$$ P(y=1\mid x) $$

This probability is then converted into a class prediction using a threshold, usually $0.5$.

The complete pipeline is:

$$ x \rightarrow z=x^TW+b \rightarrow p=\sigma(z) \rightarrow L \rightarrow \nabla L \rightarrow \text{Gradient Descent} $$

where:

- $x$ is the input feature vector.
- $W$ is the weight vector.
- $b$ is the bias.
- $z$ is the linear score.
- $p$ is the predicted probability.
- $L$ is the Binary Cross Entropy loss.

---

# 2. Linear Transformation

Suppose one sample has $d$ features:

$$ x= \begin{bmatrix} x_1\\ x_2\\ \vdots\\ x_d \end{bmatrix} $$

and the corresponding weights are:

$$ W= \begin{bmatrix} w_1\\ w_2\\ \vdots\\ w_d \end{bmatrix} $$

The model first computes a linear combination:

$$ z=x^TW+b $$

Expanding the dot product:

$$ z=x_1w_1+x_2w_2+\cdots+x_dw_d+b $$

The weights determine how strongly each feature contributes to the score, while the bias shifts the score independently of the input features.

For a dataset with $N$ samples:

$$ X= \begin{bmatrix} x_1^T\\ x_2^T\\ \vdots\\ x_N^T \end{bmatrix} $$

the scores are computed simultaneously:

$$ Z=XW+b $$

where:

$$ X\in\mathbb{R}^{N\times d} $$

$$ W\in\mathbb{R}^{d} $$

and:

$$ Z\in\mathbb{R}^{N} $$

In NumPy, this is:

```python
z = X @ W + b
```

---

# 3. Sigmoid Function

The linear score $z$ can take any real value:

$$ -\infty < z < \infty $$

A probability, however, must lie between $0$ and $1$.

Logistic Regression therefore applies the sigmoid function:

$$ \sigma(z)=\frac{1}{1+e^{-z}} $$

The model prediction is:

$$ p=\sigma(z) $$

### 3.1 Behavior of sigmoid

When $z\rightarrow+\infty$:

$$ e^{-z}\rightarrow0 $$

so:

$$ \sigma(z)\rightarrow1 $$

When $z\rightarrow-\infty$:

$$ e^{-z}\rightarrow\infty $$

so:

$$ \sigma(z)\rightarrow0 $$

At $z=0$:

$$ \sigma(0)=\frac{1}{2}=0.5 $$

Therefore:

$$ 0<\sigma(z)<1 $$

which makes the output suitable for probability interpretation.

---

# 4. Decision Boundary

The standard classification rule is:

$$ \hat{y}= \begin{cases} 1 & p\geq0.5\\ 0 & p<0.5 \end{cases} $$

Because:

$$ \sigma(0)=0.5 $$

and the sigmoid function is monotonically increasing, the threshold $p=0.5$ corresponds exactly to:

$$ z=0 $$

Therefore the decision boundary satisfies:

$$ x^TW+b=0 $$

For two features:

$$ w_1x_1+w_2x_2+b=0 $$

Solving for $x_2$:

$$ x_2=-\frac{w_1x_1+b}{w_2} $$

This is a straight line.

Thus, despite the sigmoid function being nonlinear, Logistic Regression is a **linear classifier with respect to the original input features**.

---

# 5. Binary Cross Entropy

The model outputs a probability $p$ for class $1$.

For a single example with binary label $y$, the Binary Cross Entropy loss is:

$$ L = -\left[ y\log(p)+(1-y)\log(1-p) \right] $$

## 5.1 When $y=1$

Substitute $y=1$:

$$ L = -\left[ 1\cdot\log(p)+0\cdot\log(1-p) \right] $$

Therefore:

$$ L=-\log(p) $$

If the model predicts $p$ close to $1$, the loss is small.

If the model predicts $p$ close to $0$, the loss becomes very large.

## 5.2 When $y=0$

Substitute $y=0$:

$$ L = -\left[ 0\cdot\log(p)+1\cdot\log(1-p) \right] $$

Therefore:

$$ L=-\log(1-p) $$

If $p$ is close to $0$, the loss is small.

If $p$ is close to $1$, the loss is very large.

## 5.3 Dataset loss

For $N$ samples, the mean Binary Cross Entropy is:

$$ L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i\log(p_i) + (1-y_i)\log(1-p_i) \right] $$

The implementation is:

```python
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1 - y_true) * np.log(1 - y_pred)
    )

    return loss
```

---

# 6. Binary Cross Entropy as Negative Log-Likelihood

The BCE loss is closely connected to maximum likelihood estimation.

For a binary random variable $y$, the Bernoulli probability mass function is:

$$ P(y\mid x) = p^y(1-p)^{1-y} $$

For $N$ independent samples:

$$ \mathcal{L} = \prod_{i=1}^{N} p_i^{y_i} (1-p_i)^{1-y_i} $$

Taking the logarithm:

$$ \log\mathcal{L} = \sum_{i=1}^{N} \log \left( p_i^{y_i}(1-p_i)^{1-y_i} \right) $$

Using:

$$ \log(ab)=\log(a)+\log(b) $$

gives:

$$ \log\mathcal{L} = \sum_{i=1}^{N} \left[ y_i\log(p_i) + (1-y_i)\log(1-p_i) \right] $$

Maximum likelihood seeks to maximize $\log\mathcal{L}$.

Equivalently, we minimize the negative log-likelihood:

$$ -\log\mathcal{L} = -\sum_{i=1}^{N} \left[ y_i\log(p_i) + (1-y_i)\log(1-p_i) \right] $$

Dividing by $N$ gives the mean BCE:

$$ L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i\log(p_i) + (1-y_i)\log(1-p_i) \right] $$

Thus Binary Cross Entropy can be interpreted as the average negative log-likelihood of the observed labels.

This connection will be reused when deriving multiclass cross entropy for Softmax Regression.

---

# 7. Gradient Derivation

The model is trained by minimizing the loss with respect to $W$ and $b$.

For one sample:

$$ z=x^TW+b $$

$$ p=\sigma(z) $$

$$ L = -\left[ y\log(p)+(1-y)\log(1-p) \right] $$

We will derive:

$$ \frac{\partial L}{\partial z} $$

first, and then use it to obtain the gradients with respect to $W$ and $b$.

---

## 7.1 Derivative of BCE with respect to $p$

Start from:

$$ L = -\left[ y\log(p)+(1-y)\log(1-p) \right] $$

Rewrite:

$$ L = -y\log(p) -(1-y)\log(1-p) $$

Differentiate with respect to $p$:

$$ \frac{\partial L}{\partial p} = -y\frac{1}{p} -(1-y)\frac{1}{1-p}(-1) $$

Therefore:

$$ \frac{\partial L}{\partial p} = -\frac{y}{p} + \frac{1-y}{1-p} $$

---

## 7.2 Derivative of sigmoid

The sigmoid is:

$$ p=\sigma(z)=\frac{1}{1+e^{-z}} $$

Rewrite it as:

$$ p=(1+e^{-z})^{-1} $$

Differentiate:

$$ \frac{dp}{dz} = -(1+e^{-z})^{-2}(-e^{-z}) $$

Therefore:

$$ \frac{dp}{dz} = \frac{e^{-z}}{(1+e^{-z})^2} $$

Now derive an equivalent form using $p$.

Since:

$$ p=\frac{1}{1+e^{-z}} $$

we have:

$$ 1-p = 1-\frac{1}{1+e^{-z}} $$

$$ 1-p = \frac{e^{-z}}{1+e^{-z}} $$

Multiplying $p$ and $1-p$:

$$ p(1-p) = \frac{1}{1+e^{-z}} \frac{e^{-z}}{1+e^{-z}} $$

so:

$$ p(1-p) = \frac{e^{-z}}{(1+e^{-z})^2} $$

Therefore:

$$ \boxed{ \frac{dp}{dz}=p(1-p) } $$

This compact form is especially useful for backpropagation.

---

# 8. Chain Rule: Deriving $\partial L/\partial z$

We have:

$$ L=L(p(z)) $$

so the chain rule gives:

$$ \frac{\partial L}{\partial z} = \frac{\partial L}{\partial p} \frac{\partial p}{\partial z} $$

Substitute the two derivatives:

$$ \frac{\partial L}{\partial z} = \left( -\frac{y}{p} + \frac{1-y}{1-p} \right) p(1-p) $$

Distribute:

$$ \frac{\partial L}{\partial z} = -y(1-p)+(1-y)p $$

Expand:

$$ \frac{\partial L}{\partial z} = -y+yp+p-yp $$

The two $yp$ terms cancel:

$$ \boxed{ \frac{\partial L}{\partial z}=p-y } $$

This simplification is one of the most important results in the Logistic Regression derivation.

---

# 9. Derivative with Respect to the Weights

For one sample:

$$ z=x^TW+b $$

Therefore, for weight $w_j$:

$$ \frac{\partial z}{\partial w_j}=x_j $$

Using the chain rule:

$$ \frac{\partial L}{\partial w_j} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial w_j} $$

Substitute:

$$ \frac{\partial L}{\partial z}=p-y $$

and:

$$ \frac{\partial z}{\partial w_j}=x_j $$

giving:

$$ \boxed{ \frac{\partial L}{\partial w_j} = (p-y)x_j } $$

For the entire weight vector:

$$ \boxed{ \nabla_W L=(p-y)x } $$

This is the gradient for one sample.

---

# 10. Derivative with Respect to the Bias

From:

$$ z=x^TW+b $$

we have:

$$ \frac{\partial z}{\partial b}=1 $$

Therefore:

$$ \frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \frac{\partial z}{\partial b} $$

giving:

$$ \boxed{ \frac{\partial L}{\partial b}=p-y } $$

---

# 11. Batch Gradient

The actual implementation uses the mean loss over $N$ samples:

$$ L = \frac{1}{N} \sum_{i=1}^{N}L_i $$

Therefore the gradient is the average of the individual gradients.

For each sample:

$$ \nabla_W L_i=(p_i-y_i)x_i $$

Thus:

$$ \nabla_W L = \frac{1}{N} \sum_{i=1}^{N} (p_i-y_i)x_i $$

This can be written compactly in matrix form.

Define:

$$ X= \begin{bmatrix} x_1^T\\ x_2^T\\ \vdots\\ x_N^T \end{bmatrix} $$

and:

$$ P-Y= \begin{bmatrix} p_1-y_1\\ p_2-y_2\\ \vdots\\ p_N-y_N \end{bmatrix} $$

Then:

$$ X^T(P-Y) $$

produces the sum of $(p_i-y_i)x_i$ over all samples.

Therefore:

$$ \boxed{ dW= \frac{1}{N}X^T(P-Y) } $$

Similarly, the bias gradient is:

$$ \boxed{ db= \frac{1}{N} \sum_{i=1}^{N}(p_i-y_i) } $$

or equivalently:

$$ \boxed{ db=\mathrm{mean}(P-Y) } $$

The corresponding implementation is:

```python
def compute_gradients(X, y, p):
    N = X.shape[0]
    error = p - y

    dW = (X.T @ error) / N
    db = np.mean(error)

    return dW, db
```

The mathematical correspondence is direct:

| Mathematics | NumPy |
|---|---|
| $P-Y$ | `p - y` |
| $X^T(P-Y)$ | `X.T @ error` |
| $\frac{1}{N}X^T(P-Y)$ | `(X.T @ error) / N` |
| $\frac{1}{N}\sum_i(P_i-Y_i)$ | `np.mean(error)` |

---

# 12. Backpropagation Interpretation

The gradient derivation can also be viewed as a small computational graph:

$$ x \rightarrow z \rightarrow p \rightarrow L $$

During the forward pass:

$$ z=x^TW+b $$

$$ p=\sigma(z) $$

$$ L=L_{\mathrm{BCE}}(p,y) $$

During the backward pass, the chain rule moves in the opposite direction:

$$ \frac{\partial L}{\partial p} \rightarrow \frac{\partial L}{\partial z} \rightarrow \frac{\partial L}{\partial W}, \frac{\partial L}{\partial b} $$

The key simplification is:

$$ \frac{\partial L}{\partial z}=p-y $$

which makes the implementation particularly compact.

---

# 13. Gradient Descent

Once the gradients are known, the parameters are updated using gradient descent.

The update rule for the weights is:

$$ W_{\text{new}} = W_{\text{old}} - \eta dW $$

For the bias:

$$ b_{\text{new}} = b_{\text{old}} - \eta db $$

where $\eta>0$ is the learning rate.

In implementation:

```python
def update_parameters(W, b, dW, db, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db

    return W, b
```

---

## 13.1 Why subtract the gradient?

The gradient points in the direction of greatest local increase of the loss.

Therefore the negative gradient points in the direction of greatest local decrease.

For a scalar parameter $\theta$:

$$ \theta_{\text{new}} = \theta-\eta\frac{\partial L}{\partial\theta} $$

The learning rate controls the step size.

- If $\eta$ is too small, training can be very slow.
- If $\eta$ is too large, updates can overshoot low-loss regions or become unstable.
- An appropriate value allows the loss to decrease effectively.

---

# 14. Complete Training Algorithm

Putting everything together, full-batch Logistic Regression training is:

### Initialize

$$ W=0 $$

$$ b=0 $$

### Repeat for each iteration

#### Forward pass

$$ Z=XW+b $$

$$ P=\sigma(Z) $$

#### Loss

$$ L= -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i\log(p_i) + (1-y_i)\log(1-p_i) \right] $$

#### Backward pass

$$ dW=\frac{1}{N}X^T(P-Y) $$

$$ db=\frac{1}{N}\sum_{i=1}^{N}(P_i-Y_i) $$

#### Parameter update

$$ W\leftarrow W-\eta dW $$

$$ b\leftarrow b-\eta db $$

This is repeated until the chosen number of iterations has been reached.

The implementation is:

```python
def train_logistic_regression(
    X,
    y,
    learning_rate=0.1,
    num_iterations=1000,
):
    W = np.zeros(X.shape[1])
    b = 0.0

    loss_history = []

    for _ in range(num_iterations):
        p = predict_proba(X, W, b)
        loss = binary_cross_entropy(y, p)
        dW, db = compute_gradients(X, y, p)

        W, b = update_parameters(
            W,
            b,
            dW,
            db,
            learning_rate,
        )

        loss_history.append(loss)

    return W, b, loss_history
```

---

# 15. Analytical Gradient vs Numerical Gradient

The analytical gradient was derived using calculus.

For example:

$$ \frac{\partial L}{\partial W} = \frac{1}{N}X^T(P-Y) $$

This is the expression implemented by:

```python
dW = (X.T @ (p - y)) / N
```

A gradient check provides an independent way to verify this implementation.

Instead of using the analytical formula, numerical differentiation estimates a derivative by slightly perturbing a parameter.

For parameter $w_j$, the central finite-difference approximation is:

$$ \frac{\partial L}{\partial w_j} \approx \frac{ L(w_j+\epsilon)-L(w_j-\epsilon) }{ 2\epsilon } $$

where $\epsilon$ is small.

For example, for $w_1$:

$$ \frac{\partial L}{\partial w_1} \approx \frac{ L(w_1+\epsilon)-L(w_1-\epsilon) }{ 2\epsilon } $$

This numerical estimate does not use the analytical gradient formula.

Instead:

1. Increase the parameter slightly.
2. Calculate the new loss.
3. Decrease the parameter slightly.
4. Calculate the new loss.
5. Estimate the slope from the difference.

If:

$$ \text{analytical gradient} \approx \text{numerical gradient} $$

then there is strong evidence that both the mathematical derivation and implementation are correct.

A small difference is expected because finite differences are an approximation and floating-point arithmetic introduces numerical error.

---

# 16. Numerical Gradient Check Implementation

A simple check for the first weight is:

```python
epsilon = 1e-5

W_plus = W.copy()
W_plus[0] += epsilon

W_minus = W.copy()
W_minus[0] -= epsilon

loss_plus = binary_cross_entropy(
    y,
    predict_proba(X, W_plus, b)
)

loss_minus = binary_cross_entropy(
    y,
    predict_proba(X, W_minus, b)
)

numerical_dW1 = (loss_plus - loss_minus) / (2 * epsilon)

print("Analytical dW[0]:", dW[0])
print("Numerical dW[0]: ", numerical_dW1)
print("Difference:", abs(dW[0] - numerical_dW1))
```

The purpose of this check is not to replace the derivation. It is an implementation-level verification.

---

# 17. Evaluation Metrics

After training, the model produces binary predictions:

$$ \hat y= \begin{cases} 1 & p\geq0.5\\ 0 & p<0.5 \end{cases} $$

The predictions can be evaluated against the true labels.

## 17.1 Accuracy

$$ \mathrm{Accuracy} = \frac{\text{number of correct predictions}}{N} $$

or:

$$ \mathrm{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}(\hat y_i=y_i) $$

## 17.2 Confusion Matrix

Binary classification gives four cases:

| | Actual 0 | Actual 1 |
|---|---:|---:|
| Predicted 0 | True Negative (TN) | False Negative (FN) |
| Predicted 1 | False Positive (FP) | True Positive (TP) |

## 17.3 Precision

$$ \mathrm{Precision} = \frac{TP}{TP+FP} $$

Precision asks:

> Among the samples predicted as positive, how many were actually positive?

## 17.4 Recall

$$ \mathrm{Recall} = \frac{TP}{TP+FN} $$

Recall asks:

> Among the samples that were actually positive, how many did the model identify?

## 17.5 F1 Score

$$ F1 = 2 \frac{ \mathrm{Precision}\cdot\mathrm{Recall} }{ \mathrm{Precision}+\mathrm{Recall} } $$

F1 provides a single value that balances precision and recall.

---

# 18. Module Relationships

The implementation can be understood as the following modules:

$$ \boxed{ \text{Linear Model} \rightarrow \text{Sigmoid} \rightarrow \text{BCE} \rightarrow \text{Gradient} \rightarrow \text{Gradient Descent} } $$

The training pipeline is:

$$ \text{Forward} \rightarrow \text{Loss} \rightarrow \text{Backward} \rightarrow \text{Update} \rightarrow \text{Repeat} $$

This modular structure is intentional.

When the project moves to Softmax Regression, the overall training framework can remain similar while binary-specific components are replaced.

---

# 19. Transition to Softmax Regression

Logistic Regression produces one probability for class $1$:

$$ p=P(y=1\mid x) $$

This is appropriate for:

$$ y\in\{0,1\} $$

MNIST instead contains ten classes:

$$ y\in\{0,1,\ldots,9\} $$

For multiclass classification, the model needs a probability for every class:

$$ P(y=0\mid x), P(y=1\mid x), \ldots, P(y=9\mid x) $$

These probabilities must satisfy:

$$ \sum_{k=0}^{9}P(y=k\mid x)=1 $$

This motivates the Softmax function.

For class $k$, Softmax is:

$$ \mathrm{softmax}(z)_k = \frac{e^{z_k}} {\sum_{j=1}^{K}e^{z_j}} $$

The corresponding multiclass loss is Cross Entropy:

$$ L = -\sum_{k=1}^{K} y_k\log(p_k) $$

For one-hot labels, only the true class contributes to the loss.

The important structural transition is therefore:

$$ \boxed{ \text{Sigmoid} \rightarrow \text{Softmax} } $$

and:

$$ \boxed{ \text{Binary Cross Entropy} \rightarrow \text{Multiclass Cross Entropy} } $$

The optimization principle remains gradient descent, and the overall training loop retains the same structure.

---

# 20. Summary of Key Equations

### Linear model

$$ z=x^TW+b $$

### Sigmoid

$$ p=\sigma(z)=\frac{1}{1+e^{-z}} $$

### Binary Cross Entropy

$$ L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i\log(p_i) + (1-y_i)\log(1-p_i) \right] $$

### Sigmoid derivative

$$ \frac{dp}{dz}=p(1-p) $$

### Loss derivative with respect to score

$$ \boxed{ \frac{\partial L}{\partial z}=p-y } $$

### Weight gradient

$$ \boxed{ dW=\frac{1}{N}X^T(P-Y) } $$

### Bias gradient

$$ \boxed{ db=\frac{1}{N}\sum_{i=1}^{N}(P_i-Y_i) } $$

### Gradient descent

$$ \boxed{ W\leftarrow W-\eta dW } $$

$$ \boxed{ b\leftarrow b-\eta db } $$

### Binary prediction

$$ \hat y= \begin{cases} 1 & p\geq0.5\\ 0 & p<0.5 \end{cases} $$

### Decision boundary

$$ \boxed{ x^TW+b=0 } $$
