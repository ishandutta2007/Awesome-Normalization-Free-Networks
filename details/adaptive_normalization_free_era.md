# The Adaptive Normalization-Free Era

NFNets achieved absolute scale-invariant optimization without any normalization layers by utilizing Adaptive Gradient Clipping (AGC) and specifically scaled residual branches.

```mermaid
flowchart LR
    A[Calculate Gradients] --> B[Compare to Weight Magnitude]
    B --> C[Clip Gradients Dynamically]
    C --> D[Update Weights safely]
```

[Back to README](../README.md)
