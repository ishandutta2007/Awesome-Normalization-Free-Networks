import os

details_dir = "details"
os.makedirs(details_dir, exist_ok=True)

pages = {
    "un_normalized_exploding_era.md": {
        "title": "The Un-normalized Exploding Era",
        "content": "The un-normalized exploding era of CNNs refers to the early days of deep learning where scaling deep neural networks caused massive instability. Without normalization layers, backpropagated gradients would either vanish or explode.\n\n```mermaid\nflowchart TD\n    A[Input Data] --> B[Conv Layers]\n    B --> C{Exploding / Vanishing Gradients}\n    C -->|Explode| D[NaN Weights]\n    C -->|Vanish| E[Zero Learning]\n```"
    },
    "batch_normalization_hegemony.md": {
        "title": "The Batch Normalization Hegemony",
        "content": "Batch Normalization revolutionized deep network training by standardizing activations along the mini-batch dimension. This solved the vanishing/exploding gradient problem but introduced batch-size dependencies.\n\n```mermaid\nflowchart LR\n    A[Input Batch] --> B[Compute Mean & Var]\n    B --> C[Normalize]\n    C --> D[Scale & Shift]\n```"
    },
    "weight_standardization_bit.md": {
        "title": "Weight Standardization & BiT",
        "content": "Weight Standardization shifted the normalization focus from activations to the model weights themselves, avoiding batch dimension dependencies.\n\n```mermaid\nflowchart TD\n    A[Original Weights] --> B[Subtract Mean]\n    B --> C[Divide by Variance]\n    C --> D[Standardized Weights]\n```"
    },
    "adaptive_normalization_free_era.md": {
        "title": "The Adaptive Normalization-Free Era",
        "content": "NFNets achieved absolute scale-invariant optimization without any normalization layers by utilizing Adaptive Gradient Clipping (AGC) and specifically scaled residual branches.\n\n```mermaid\nflowchart LR\n    A[Calculate Gradients] --> B[Compare to Weight Magnitude]\n    B --> C[Clip Gradients Dynamically]\n    C --> D[Update Weights safely]\n```"
    },
    "nf_resnets.md": {
        "title": "NF-ResNets",
        "content": "NF-ResNets represent an early prototype of Normalization-Free networks, directly modifying classic ResNets with specific variance multipliers.\n\n```mermaid\nflowchart TB\n    A[Input x_l] --> B[Residual Path: f(x_l / beta)]\n    A --> C[Identity Path]\n    B --> D[Multiply by alpha]\n    C --> E[Add]\n    D --> E\n    E --> F[Output x_l+1]\n```"
    },
    "core_nfnets.md": {
        "title": "Core NFNets (AGC Class)",
        "content": "Core NFNets use Scaled Weight Standardization along with Adaptive Gradient Clipping to maintain stability at very large scale training.\n\n```mermaid\nflowchart TD\n    A[Forward: Scaled WS] --> B[Loss Calculation]\n    B --> C[Backward: Compute G]\n    C --> D[AGC Clip threshold eval]\n    D --> E[Update Weights]\n```"
    },
    "nf_vits.md": {
        "title": "NF-Vision Transformers (NF-ViTs)",
        "content": "NF-ViTs apply normalization-free principles to Vision Transformers, removing LayerNorms completely.\n\n```mermaid\nflowchart LR\n    A[Tokens] --> B[Attention without LayerNorm]\n    B --> C[Scaled Initializations]\n    C --> D[Stable Output Variance]\n```"
    },
    "scaled_weight_standardization.md": {
        "title": "Scaled Weight Standardization",
        "content": "It calibrates the kernel variance. It modifies convolutional weights before math steps, giving zero mean and tailored variance.\n\n```mermaid\nflowchart LR\n    A[Raw Kernel] --> B[Zero-Mean]\n    B --> C[Scale Variance to match Activation Func]\n    C --> D[Execute Conv]\n```"
    },
    "lambda_clipping_threshold.md": {
        "title": "The λ-Clipping Threshold",
        "content": "This hyperparameter regulates the max allowable gradient-to-weight ratio, binding step sizes safely.\n\n```mermaid\nflowchart TD\n    A[Compute Layer Gradient Norm] --> B{Ratio > Lambda?}\n    B -->|Yes| C[Clip Gradient]\n    B -->|No| D[Keep Gradient]\n```"
    },
    "memory_allocation_gap.md": {
        "title": "High-Throughput Memory Allocation Gap",
        "content": "Calculating Frobenius norms adds processing overhead. Fused kernels help bypass this.\n\n```mermaid\nflowchart LR\n    A[Gradient Calculation] --> B[Fused Triton/CUDA Kernel]\n    B --> C[Registers (Fast SRAM)]\n    C --> D[Updated Weights]\n```"
    },
    "mixed_precision_underflow.md": {
        "title": "Low-Precision Underflow Hazard",
        "content": "Terminal layer gradients might underflow in FP16. Maintaining an FP32 Master Weight fixes this.\n\n```mermaid\nflowchart TB\n    A[FP16 Forward & Backward] --> B[Compute Gradients]\n    B --> C[Cast to FP32]\n    C --> D[Update FP32 Master Weights]\n```"
    },
    "cloud_vision_serving.md": {
        "title": "Cloud Vision Serving",
        "content": "NFNets handle varying inference batch sizes flawlessly in high-volume enterprise systems.\n\n```mermaid\nflowchart LR\n    A[Varying Traffic] --> B[Dynamic Batching]\n    B --> C[NFNet Model (Stable)]\n    C --> D[Inference Output]\n```"
    },
    "autonomous_driving.md": {
        "title": "Autonomous Driving (BEV)",
        "content": "Zero batch-dependency allows decoupled spatial feature extraction for fleets without node syncing.\n\n```mermaid\nflowchart TD\n    A[Camera Streams] --> B[Async GPU Nodes]\n    A --> C[LiDAR Grids]\n    B --> D[Real-time Tracking]\n    C --> D\n```"
    },
    "medical_imaging.md": {
        "title": "Medical Volumetric Tracking",
        "content": "For huge volumetric inputs where batch size is 1 or 2, NFNets offer extreme stability without BatchNorm breakage.\n\n```mermaid\nflowchart LR\n    A[Multi-Megapixel Scans] --> B[Batch Size = 1]\n    B --> C[NFNet Backbone]\n    C --> D[Diagnostic Precision]\n```"
    }
}

for filename, data in pages.items():
    filepath = os.path.join(details_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {data['title']}\n\n")
        f.write(f"{data['content']}\n\n")
        f.write(f"[Back to README](../README.md)\n")
