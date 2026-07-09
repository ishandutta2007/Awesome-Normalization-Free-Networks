# NF-Vision Transformers (NF-ViTs)

NF-ViTs apply normalization-free principles to Vision Transformers, removing LayerNorms completely.

```mermaid
flowchart LR
    A[Tokens] --> B[Attention without LayerNorm]
    B --> C[Scaled Initializations]
    C --> D[Stable Output Variance]
```

[Back to README](../README.md)
