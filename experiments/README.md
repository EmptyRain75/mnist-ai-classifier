# Experiments

This folder contains structured experiments used to evaluate and compare the models developed in this project.

The notebooks in `notebooks/` are primarily used for learning, implementation, and incremental experimentation. Because those experiments are performed while developing each model, they may be exploratory, incomplete, or focused on only one concept at a time or just utter trash.

The purpose of this folder is different.

```text
notebooks/
Learning
Implementation
Small exploratory experiments
Step-by-step analysis

        ↓

experiments/
Systematic evaluation
Controlled comparisons
Broader model analysis
Final experiment results
```

The experiments in this folder are intended to provide a more complete picture of model behavior.

## Goals

The main goals are to:

- Compare different model architectures fairly.
- Test important hyperparameters systematically.
- Study training and validation behavior.
- Measure model capacity and computational cost.
- Analyze generalization and overfitting.
- Compare optimization methods where appropriate.
- Study model errors and confusion patterns.
- Produce plots and tables suitable for final project analysis.
- Validate conclusions made during the learning notebooks.

## Experimental Principles

Experiments should follow several basic rules.

### Controlled comparisons

When comparing one factor, other important settings should remain fixed whenever possible.

For example:

```text
Experiment:
Effect of hidden size

Change:
Hidden size

Keep fixed:
Dataset
Train/validation split
Batch size
Learning rate
Optimizer
Number of epochs
Random seed
```

This makes it easier to understand what caused the observed difference.

### Reproducibility

Experiments should use fixed random seeds where possible.

Important settings such as:

```text
Random seed
Learning rate
Batch size
Epochs
Optimizer
Architecture
Dataset split
```

should be recorded with the results.

### Report observed results

Experiments should report what actually happens rather than being designed to force an expected conclusion.

Unexpected or negative results are still useful because they help reveal the limitations and behavior of the models.

### Validation before test evaluation

The validation set should be used for model and hyperparameter selection.

The test set should be used only for final evaluation after the experiment configuration has been selected.

## Planned Experiment Categories

### Model Capacity

Examples:

```text
Hidden-layer width
Network depth
Number of parameters
```

### Optimization

Examples:

```text
Learning rate
SGD
Momentum
Adam
Learning-rate scheduling
```

### Training Configuration

Examples:

```text
Batch size
Number of epochs
Weight initialization
```

### Generalization

Examples:

```text
Training vs validation accuracy
Training vs validation loss
Overfitting behavior
Regularization
```

### Architecture Comparison

The main model progression can be compared directly:

```text
Softmax Regression
        ↓
MLP
        ↓
CNN
        ↓
Vision Transformer
```

Possible comparison metrics include:

```text
Validation accuracy
Test accuracy
Training loss
Validation loss
Number of parameters
Training time
Inference time
```

### Error Analysis

Experiments may also investigate:

```text
Confusion matrix
Frequently confused classes
Misclassified examples
Prediction confidence
```

## Suggested Structure

As the project grows, experiments may be organized into subfolders:

```text
experiments/
│
├── README.md
│
├── softmax/
│
├── mlp/
│
├── cnn/
│
├── vit/
│
└── comparisons/
```

Possible contents include:

```text
.ipynb experiment notebooks
.csv result tables
.png figures
.md experiment summaries
```

The exact structure may evolve as the number of experiments increases.

## Relationship to the Rest of the Repository

```text
docs/
Deep theory and mathematical explanations

notebooks/
Learning, implementation, and exploratory analysis

src/
Clean reusable model implementations

experiments/
Systematic experiments and model comparisons
```

The `experiments/` folder therefore acts as the main location for studying model behavior beyond the smaller experiments performed during the learning process.
