# High-Throughput Memory Allocation Gap

Calculating Frobenius norms adds processing overhead. Fused kernels help bypass this.

```mermaid
flowchart LR
    A[Gradient Calculation] --> B[Fused Triton/CUDA Kernel]
    B --> C[Registers (Fast SRAM)]
    C --> D[Updated Weights]
```

[Back to README](../README.md)
