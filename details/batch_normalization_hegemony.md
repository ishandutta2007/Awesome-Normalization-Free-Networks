# The Batch Normalization Hegemony

Batch Normalization revolutionized deep network training by standardizing activations along the mini-batch dimension. This solved the vanishing/exploding gradient problem but introduced batch-size dependencies.

```mermaid
flowchart LR
    A[Input Batch] --> B[Compute Mean & Var]
    B --> C[Normalize]
    C --> D[Scale & Shift]
```

[Back to README](../README.md)
