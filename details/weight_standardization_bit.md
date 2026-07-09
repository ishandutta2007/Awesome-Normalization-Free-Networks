# Weight Standardization & BiT

Weight Standardization shifted the normalization focus from activations to the model weights themselves, avoiding batch dimension dependencies.

```mermaid
flowchart TD
    A[Original Weights] --> B[Subtract Mean]
    B --> C[Divide by Variance]
    C --> D[Standardized Weights]
```

[Back to README](../README.md)
