import re

def process_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # Section 1
    s1_target = """*   **The Un-normalized Exploding Era (Traditional ConvNets, Pre-2015)**
    *   *Concept:* The early deep learning baseline (e.g., VGG, AlexNet). Scaling neural networks past a depth of ~20 layers caused backpropagated error signals to decay or explode exponentially as they traveled through deep sequential matrices, stalling optimization or triggering numerical overflow.
*   **The Batch Normalization Hegemony (Vanilla ResNet / ConvNets, 2015–2020)**
    *   *Concept:* Dismantled the early depth wall by introducing **Batch Normalization (BatchNorm)**. It stabilized activation variance symmetrically by calculating the running mean and variance of features across the current mini-batch during every forward pass.
    *   *Limitation:* Catastrophic multi-node memory chokes and batch-size dependencies. BatchNorm breaks when batch sizes drop too low (e.g., when training on high-resolution images or videos where only 1 or 2 frames fit in VRAM). Furthermore, in distributed clusters, it forces nodes to execute costly intra-network synchronization steps to share batch statistics, choking communication bandwidth.
*   **The Weight Standardization & Layer-Norm Alternatives (Big Transfer / BiT, ~2019–2020)**
    *   *Concept:* Attempted to remove batch dependencies by shifting normalization from *data activations* straight to the *model weights* themselves. Frameworks like Google's **Big Transfer (BiT)** combined **Weight Standardization (WS)**—re-centering and scaling the convolutional kernel parameters to zero mean and unit variance—with **Group Normalization (GroupNorm)**.
    *   *Limitation:* Retained an explicit normalization math overhead step, which added calculation latency and slightly degraded parameter expressiveness on web-scale datasets.
*   **The Adaptive Normalization-Free Era (NFNets, Brock et al. / DeepMind, 2021–Present)**
    *   *Concept:* The current modern state-of-the-art vision infrastructure standard. Developed by Andrew Brock, Soham De, Samuel L. Smith, and Volodya Razavi, **NFNets** eliminated normalization layers completely. It achieved absolute scale-invariant optimization by introducing **Adaptive Gradient Clipping (AGC)**.
    *   *Significance:* AGC monitors and clips layer gradients relative to the magnitude of that layer's existing weights. Combined with **Scaled Weight Standardization** and specialized initialization multipliers, NFNets train stably at massive scales, removing multi-node synchronization stalls entirely to deliver rapid training velocities on modern GPU clusters."""
    s1_repl = """| Era | Concept & Limitation | Year | Paper Link | Details |
|---|---|---|---|---|
| **The Un-normalized Exploding Era (Traditional ConvNets, Pre-2015)** | **Concept:** The early deep learning baseline (e.g., VGG, AlexNet). Scaling neural networks past a depth of ~20 layers caused backpropagated error signals to decay or explode exponentially as they traveled through deep sequential matrices, stalling optimization or triggering numerical overflow. | Pre-2015 | [AlexNet (2012)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) | [Read More](details/un_normalized_exploding_era.md) |
| **The Batch Normalization Hegemony (Vanilla ResNet / ConvNets, 2015–2020)** | **Concept:** Dismantled the early depth wall by introducing **Batch Normalization (BatchNorm)**. It stabilized activation variance symmetrically by calculating the running mean and variance of features across the current mini-batch during every forward pass.<br><br>**Limitation:** Catastrophic multi-node memory chokes and batch-size dependencies. BatchNorm breaks when batch sizes drop too low (e.g., when training on high-resolution images or videos where only 1 or 2 frames fit in VRAM). Furthermore, in distributed clusters, it forces nodes to execute costly intra-network synchronization steps to share batch statistics, choking communication bandwidth. | 2015 | [Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167) | [Read More](details/batch_normalization_hegemony.md) |
| **The Weight Standardization & Layer-Norm Alternatives (Big Transfer / BiT, ~2019–2020)** | **Concept:** Attempted to remove batch dependencies by shifting normalization from *data activations* straight to the *model weights* themselves. Frameworks like Google's **Big Transfer (BiT)** combined **Weight Standardization (WS)**—re-centering and scaling the convolutional kernel parameters to zero mean and unit variance—with **Group Normalization (GroupNorm)**.<br><br>**Limitation:** Retained an explicit normalization math overhead step, which added calculation latency and slightly degraded parameter expressiveness on web-scale datasets. | 2019 | [Qiao et al. (2019)](https://arxiv.org/abs/1903.10520) | [Read More](details/weight_standardization_bit.md) |
| **The Adaptive Normalization-Free Era (NFNets, Brock et al. / DeepMind, 2021–Present)** | **Concept:** The current modern state-of-the-art vision infrastructure standard. Developed by Andrew Brock, Soham De, Samuel L. Smith, and Volodya Razavi, **NFNets** eliminated normalization layers completely. It achieved absolute scale-invariant optimization by introducing **Adaptive Gradient Clipping (AGC)**.<br><br>**Significance:** AGC monitors and clips layer gradients relative to the magnitude of that layer's existing weights. Combined with **Scaled Weight Standardization** and specialized initialization multipliers, NFNets train stably at massive scales, removing multi-node synchronization stalls entirely to deliver rapid training velocities on modern GPU clusters. | 2021 | [Brock et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/adaptive_normalization_free_era.md) |"""

    content = content.replace(s1_target, s1_repl)

    # Section 2
    s2_target = """- ### A. NF-ResNets (Early Prototyping)
	*   **Mechanism:** Implements a direct re-parameterization of classic ResNet backbones. It replaces BatchNorm layers with explicit, fixed scalar variance-preservation multipliers ($\\alpha, \\beta$) across the residual addition highways:
	    $$x_{\\ell+1} = x_\\ell + \\alpha \\cdot f\\left(\\frac{x_\\ell}{\\beta_\\ell}\\right)$$
	    Where $\\beta_\\ell = \\sqrt{\\text{Var}(x_\\ell)}$ is calculated analytically based on prior layer initializations.

- ### B. Core NFNets (AGC Class)
	*   **Mechanism:** Blends Scaled Weight Standardization with **Adaptive Gradient Clipping (AGC)**. For layer $l$ with weights $W_l$ and gradients $G_l$, the clipping threshold is evaluated dynamically based on the Frobenius norm ($\\|\\cdot\\|_F$):
	    $$G_l \\leftarrow G_l \\times \\frac{\\lambda \\frac{\\|W_l\\|_F^*}{\\|G_l\\|_F}}{\\max\\left(1, \\lambda \\frac{\\|W_l\\|_F^*}{\\|G_l\\|_F}\\right)}, \\quad \\text{where} \\quad \\|W_l\\|_F^* = \\max(\\|W_l\\|_F, \\epsilon)$$
	*   **Behavior:** Directly restricts the gradient step size to prevent it from destabilizing large weight matrices, keeping updates bounded.

- ### C. NF-Vision Transformers (NF-ViTs)
	*   **Mechanism:** Ports normalization-free principles straight into attention-based architectures. It removes Pre-LayerNorm structures from the self-attention blocks, utilizing depth-scaled initializers ($1/\\sqrt{2L}$) to ensure gradient vectors maintain steady variances over long sequence token arrays."""
    
    s2_repl = """| Variant | Mechanism & Behavior | Year | Paper Link | Details |
|---|---|---|---|---|
| **A. NF-ResNets (Early Prototyping)** | **Mechanism:** Implements a direct re-parameterization of classic ResNet backbones. It replaces BatchNorm layers with explicit, fixed scalar variance-preservation multipliers ($\\alpha, \\beta$) across the residual addition highways:<br><br>$$x_{\\ell+1} = x_\\ell + \\alpha \\cdot f\\left(\\frac{x_\\ell}{\\beta_\\ell}\\right)$$<br><br>Where $\\beta_\\ell = \\sqrt{\\text{Var}(x_\\ell)}$ is calculated analytically based on prior layer initializations. | 2021 | [Brock et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/nf_resnets.md) |
| **B. Core NFNets (AGC Class)** | **Mechanism:** Blends Scaled Weight Standardization with **Adaptive Gradient Clipping (AGC)**. For layer $l$ with weights $W_l$ and gradients $G_l$, the clipping threshold is evaluated dynamically based on the Frobenius norm ($\\|\\cdot\\|_F$):<br><br>$$G_l \\leftarrow G_l \\times \\frac{\\lambda \\frac{\\|W_l\\|_F^*}{\\|G_l\\|_F}}{\\max\\left(1, \\lambda \\frac{\\|W_l\\|_F^*}{\\|G_l\\|_F}\\right)}, \\quad \\text{where} \\quad \\|W_l\\|_F^* = \\max(\\|W_l\\|_F, \\epsilon)$$<br><br>**Behavior:** Directly restricts the gradient step size to prevent it from destabilizing large weight matrices, keeping updates bounded. | 2021 | [De et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/core_nfnets.md) |
| **C. NF-Vision Transformers (NF-ViTs)** | **Mechanism:** Ports normalization-free principles straight into attention-based architectures. It removes Pre-LayerNorm structures from the self-attention blocks, utilizing depth-scaled initializers ($1/\\sqrt{2L}$) to ensure gradient vectors maintain steady variances over long sequence token arrays. | 2021 | [Brock et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/nf_vits.md) |"""

    content = content.replace(s2_target, s2_repl)

    # Section 3
    s3_target = """*   **Scaled Weight Standardization**
    *   *Profile:* Calibrates kernel variance. It modifies the standard convolutional weights ($W_{i,j}$) programmatically before executing any linear algebra steps, forcing zero mean and scaling the variance to precisely match the target activation function's geometry (e.g., GELU or SiLU profiles).
*   **The $\\lambda$-Clipping Threshold Dial**
    *   *Profile:* Hyperparameter optimization. The parameter $\\lambda$ regulates the maximum allowable gradient-to-weight ratio (typically locked at $\\lambda = 0.016$). Setting this value bounds step sizes, allowing the model to train cleanly with giant mini-batch sizes (>4096) without experiencing initialization-stage optimization divergence."""
    
    s3_repl = """| Component | Profile | Year | Paper Link | Details |
|---|---|---|---|---|
| **Scaled Weight Standardization** | **Profile:** Calibrates kernel variance. It modifies the standard convolutional weights ($W_{i,j}$) programmatically before executing any linear algebra steps, forcing zero mean and scaling the variance to precisely match the target activation function's geometry (e.g., GELU or SiLU profiles). | 2021 | [Brock et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/scaled_weight_standardization.md) |
| **The $\\lambda$-Clipping Threshold Dial** | **Profile:** Hyperparameter optimization. The parameter $\\lambda$ regulates the maximum allowable gradient-to-weight ratio (typically locked at $\\lambda = 0.016$). Setting this value bounds step sizes, allowing the model to train cleanly with giant mini-batch sizes (>4096) without experiencing initialization-stage optimization divergence. | 2021 | [De et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/lambda_clipping_threshold.md) |"""
    content = content.replace(s3_target, s3_repl)

    # Section 4
    s4_target = """*   **The High-Throughput Autoregressive Memory Allocation Gap**
    *   *The Problem:* While NFNets eliminate the memory-synchronization stalls of BatchNorm, calculating the Frobenius norms of weights and gradients for *every individual layer* during the backward pass can introduce minor software processing overheads, slightly dragging down raw matrix processing throughput.
    *   *Mitigation:* Compiling the AGC norm calculations, clipping thresholds, and parameter updates straight into a single **fused Triton or CUDA kernel execution block**, performing the math entirely within fast, on-chip GPU SRAM registers to bypass global memory bus latency.
*   **The Low-Precision Mixed-Precision Underflow Hazard**
    *   *The Problem:* Training deep NFNets using low-precision 16-bit floats (FP16 or BF16) can cause the heavily downscaled gradients of terminal layers to drop beneath numerical boundaries. This triggers numerical **underflow errors**, zeroing out learning increments and stalling loss convergence.
    *   *Mitigation:* Maintaining a strict **FP32 Master Weight Optimizer configuration (AdamW integration)**. While forward and backward passes execute in fast, low-bit 16-bit matrices, the system caches and updates a copy of the master parameters in full 32-bit floating-point registers to protect low-bit learning increments safely."""
    s4_repl = """| Challenge | Problem & Mitigation | Year | Paper Link | Details |
|---|---|---|---|---|
| **The High-Throughput Autoregressive Memory Allocation Gap** | **The Problem:** While NFNets eliminate the memory-synchronization stalls of BatchNorm, calculating the Frobenius norms of weights and gradients for *every individual layer* during the backward pass can introduce minor software processing overheads, slightly dragging down raw matrix processing throughput.<br><br>**Mitigation:** Compiling the AGC norm calculations, clipping thresholds, and parameter updates straight into a single **fused Triton or CUDA kernel execution block**, performing the math entirely within fast, on-chip GPU SRAM registers to bypass global memory bus latency. | 2021 | [Brock et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/memory_allocation_gap.md) |
| **The Low-Precision Mixed-Precision Underflow Hazard** | **The Problem:** Training deep NFNets using low-precision 16-bit floats (FP16 or BF16) can cause the heavily downscaled gradients of terminal layers to drop beneath numerical boundaries. This triggers numerical **underflow errors**, zeroing out learning increments and stalling loss convergence.<br><br>**Mitigation:** Maintaining a strict **FP32 Master Weight Optimizer configuration (AdamW integration)**. While forward and backward passes execute in fast, low-bit 16-bit matrices, the system caches and updates a copy of the master parameters in full 32-bit floating-point registers to protect low-bit learning increments safely. | 2021 | [Brock et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/mixed_precision_underflow.md) |"""
    content = content.replace(s4_target, s4_repl)

    # Section 5
    s5_target = """*   **High-Volume Real-Time Cloud Vision Serving Platforms**
    *   *Application:* Optimizes cloud serving frameworks for high-volume concurrent streams. Because NFNets possess zero batch-size dependencies, an enterprise server can execute model inference over highly fluid batch sizes (e.g., dynamically changing from batch size 1 to batch size 1024 based on live traffic fluctuations) without experiencing the performance degradation typical of BatchNorm architectures.
*   **Autonomous Driving Perception Fleet Perimeters (BEV perception)**
    *   *Application:* Coordinates real-time navigation pipelines for advanced self-driving automotive fleets. High-speed normalization-free convolutional backbones process high-resolution streaming video and LiDAR grids concurrently; the absence of BatchNorm layers allows independent GPU nodes to shard and calculate spatial features asynchronously, compressing processing latency to ensure safe real-time obstacle tracking.
*   **High-Resolution Clinical Diagnostic Volumetric Tracking (MedTech)**
    *   *Application:* Ingests massive multi-megapixel data matrices (such as MRIs, CT volumes, and digital pathology slides). Because these immense visual files consume massive VRAM, models can only process a mini-batch size of 1 or 2 samples per card. Normalization-free architectures stabilize the learning parameters under these ultra-low batch constraints, ensuring high-fidelity diagnostic precision."""
    s5_repl = """| Application | Description | Year | Paper Link | Details |
|---|---|---|---|---|
| **High-Volume Real-Time Cloud Vision Serving Platforms** | **Application:** Optimizes cloud serving frameworks for high-volume concurrent streams. Because NFNets possess zero batch-size dependencies, an enterprise server can execute model inference over highly fluid batch sizes (e.g., dynamically changing from batch size 1 to batch size 1024 based on live traffic fluctuations) without experiencing the performance degradation typical of BatchNorm architectures. | 2021 | [Brock et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/cloud_vision_serving.md) |
| **Autonomous Driving Perception Fleet Perimeters (BEV perception)** | **Application:** Coordinates real-time navigation pipelines for advanced self-driving automotive fleets. High-speed normalization-free convolutional backbones process high-resolution streaming video and LiDAR grids concurrently; the absence of BatchNorm layers allows independent GPU nodes to shard and calculate spatial features asynchronously, compressing processing latency to ensure safe real-time obstacle tracking. | 2021 | [Brock et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/autonomous_driving.md) |
| **High-Resolution Clinical Diagnostic Volumetric Tracking (MedTech)** | **Application:** Ingests massive multi-megapixel data matrices (such as MRIs, CT volumes, and digital pathology slides). Because these immense visual files consume massive VRAM, models can only process a mini-batch size of 1 or 2 samples per card. Normalization-free architectures stabilize the learning parameters under these ultra-low batch constraints, ensuring high-fidelity diagnostic precision. | 2021 | [Brock et al. (2021)](https://arxiv.org/abs/2102.06171) | [Read More](details/medical_imaging.md) |"""
    content = content.replace(s5_target, s5_repl)

    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)

process_readme()
