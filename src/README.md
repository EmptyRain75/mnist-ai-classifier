# Source Code

This folder contains the clean, reusable, and directly executable implementations extracted from the learning notebooks.

Each file is designed to:

```text
Run directly
   ↓
Load data
   ↓
Train model
   ↓
Evaluate
   ↓
Print results
```

The goal of `src/` is to keep only the essential:

- model definition
- data loading
- training
- evaluation
- small supporting utilities
- executable `main()` pipeline

Notebook explanations, visualizations, exploratory code, and experiments are kept outside this folder.

The workflow is:

```text
Notebook
   ↓
Learn and experiment
   ↓
Understand the model
   ↓
Extract clean implementation
   ↓
src/
```

The files here represent the simplified reference implementation of each model after it has been developed and tested.
