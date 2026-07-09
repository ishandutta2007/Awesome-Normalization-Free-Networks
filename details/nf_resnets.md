# NF-ResNets

NF-ResNets represent an early prototype of Normalization-Free networks, directly modifying classic ResNets with specific variance multipliers.

```mermaid
flowchart TB
    A[Input x_l] --> B[Residual Path: f(x_l / beta)]
    A --> C[Identity Path]
    B --> D[Multiply by alpha]
    C --> E[Add]
    D --> E
    E --> F[Output x_l+1]
```

[Back to README](../README.md)
