# Restormer-Lite Architecture Documentation

## Technical Documentation for Low-Light Image Denoising and Super-Resolution
 

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Architecture Design](#3-architecture-design)
4. [Component Details](#4-component-details)
5. [Data Flow](#5-data-flow)
6. [Training Configuration](#6-training-configuration)
7. [Implementation Code](#7-implementation-code)
8. [Performance Metrics](#8-performance-metrics)
9. [Comparison with Other Approaches](#9-comparison-with-other-approaches)
10. [References](#10-references)

---

## 1. Overview

Restormer-Lite is a lightweight deep learning architecture designed for joint **image denoising** and **4× super-resolution** of low-light images. The model is inspired by the original Restormer paper but optimized for computational efficiency while maintaining high restoration quality.

### Key Features

- **Lightweight Design:** Reduced parameter count for efficient inference
- **Large Kernel Attention (LKA):** Captures long-range dependencies without quadratic complexity
- **Progressive Upsampling:** PixelShuffle-based 4× super-resolution
- **End-to-End Training:** Joint optimization for denoising and super-resolution

### Input/Output Specification

| Specification | Value |
|---------------|-------|
| Input Size | 128 × 80 × 3 (RGB) |
| Output Size | 512 × 320 × 3 (RGB) |
| Scale Factor | 4× |
| Channel Dimension | 32 |

---

## 2. Problem Statement

### Challenge

Low-light images suffer from multiple degradations:
- **Noise:** High ISO settings introduce sensor noise
- **Low Resolution:** Limited detail in dark regions
- **Poor Contrast:** Reduced dynamic range

### Objective

Design a unified model that simultaneously:
1. Removes noise from low-light images
2. Upscales resolution by 4× factor
3. Preserves fine details and textures
4. Maintains computational efficiency

### Dataset

- **Source:** Kaggle Competition (dlp-jan-2025-nppe-3)
- **Training Pairs:** Low-light input → High-quality ground truth
- **Resolution:** 128×80 (input) → 512×320 (target)

---

## 3. Architecture Design

### 3.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESTORMER-LITE PIPELINE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input Image (128×80×3)                                        │
│         │                                                       │
│         ▼                                                       │
│   ┌─────────────────┐                                           │
│   │ Shallow Conv3×3 │  3 → 32 channels                          │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────────────────────────┐                       │
│   │     TransformerBlockLite × 3        │                       │
│   │  ┌─────────────────────────────┐    │                       │
│   │  │ BatchNorm → LKA → Residual  │    │                       │
│   │  │ BatchNorm → FFN → Residual  │    │                       │
│   │  └─────────────────────────────┘    │                       │
│   └────────┬────────────────────────────┘                       │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────────────────────────┐                       │
│   │   PixelShuffle Upsampling (4×)      │                       │
│   │   Conv→PS(2×)→Conv→PS(2×)           │                       │
│   └────────┬────────────────────────────┘                       │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                           │
│   │ Output Conv3×3  │  32 → 3 channels                          │
│   └────────┬────────┘                                           │
│            │                                                    │
│            ▼                                                    │
│   Output Image (512×320×3)                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Overview

| Component | Purpose | Input Shape | Output Shape |
|-----------|---------|-------------|--------------|
| Shallow Feature Extraction | Initial feature embedding | (B, 3, 128, 80) | (B, 32, 128, 80) |
| Transformer Encoder (×3) | Feature refinement with attention | (B, 32, 128, 80) | (B, 32, 128, 80) |
| PixelShuffle Upsampling | 4× spatial upscaling | (B, 32, 128, 80) | (B, 32, 512, 320) |
| Output Reconstruction | RGB image generation | (B, 32, 512, 320) | (B, 3, 512, 320) |

---

## 4. Component Details

### 4.1 Shallow Feature Extraction

**Purpose:** Convert raw RGB pixels into a learnable feature space.

```python
self.shallow_feat = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
```

**Specifications:**
- Kernel Size: 3×3
- Stride: 1
- Padding: 1 (same padding)
- Input Channels: 3 (RGB)
- Output Channels: 32

**Rationale:** A single convolutional layer efficiently projects the input into a higher-dimensional feature space while preserving spatial resolution.

---

### 4.2 Large Kernel Attention (LKA)

**Purpose:** Capture long-range spatial dependencies efficiently.

#### Architecture Diagram

```
Input Feature Map (32×H×W)
        │
        ▼
┌───────────────────────────────────┐
│  Conv 5×5 (depthwise, groups=32)  │  → Local pattern extraction
│  Receptive Field: 5×5             │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  Conv 7×7 (dilated=3, depthwise)  │  → Long-range dependencies
│  Effective Receptive Field: 21×21 │
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│  Conv 1×1 (pointwise)             │  → Channel-wise mixing
└───────────────────────────────────┘
        │
        ▼
    Attention Map
        │
        ▼
   Input × Attention  → Attended Features (element-wise gating)
```

#### Implementation

```python
class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Depthwise 5×5 convolution for local features
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        
        # Dilated depthwise 7×7 convolution for long-range features
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim)
        
        # Pointwise convolution for channel mixing
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x                      # Store original input for residual
        attn = self.conv1(x)       # Local features
        attn = self.conv2(attn)    # Long-range features
        attn = self.conv3(attn)    # Channel mixing
        return u * attn            # Element-wise attention (gating)
```

#### LKA vs Standard Self-Attention

| Aspect | Self-Attention | LKA |
|--------|----------------|-----|
| Computational Complexity | O(n²) | O(n) |
| Memory Usage | High | Low |
| Long-range Modeling | Global | Large receptive field via dilation |
| Parameter Count | High | Low (depthwise convolutions) |
| Suitable for | Small images | High-resolution images |

#### Receptive Field Calculation

For the dilated 7×7 convolution with dilation=3:
```
Effective Receptive Field = kernel_size + (kernel_size - 1) × (dilation - 1)
                         = 7 + (7 - 1) × (3 - 1)
                         = 7 + 12 = 19 pixels (per direction)
                         ≈ 21×21 effective receptive field
```

---

### 4.3 FeedForward Network (FFN)

**Purpose:** Apply non-linear transformations after attention.

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1),    # Expand: 32 → 64
            nn.GELU(),                                     # Smooth non-linearity
            nn.Conv2d(hidden_dim, dim, kernel_size=1)     # Project: 64 → 32
        )

    def forward(self, x):
        return self.net(x)
```

**Design Choices:**
- **Expansion Ratio:** 2× (32 → 64 → 32)
- **Activation:** GELU (smoother than ReLU, better gradients)
- **Implementation:** 1×1 convolutions (equivalent to MLP on spatial locations)

---

### 4.4 Transformer Block Lite

**Purpose:** Core building block combining attention and feedforward processing.

```python
class TransformerBlockLite(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = LKA(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = FeedForward(dim, ff_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))   # Pre-norm + LKA + Residual
        x = x + self.ffn(self.norm2(x))    # Pre-norm + FFN + Residual
        return x
```

#### Block Diagram

```
        Input (x)
           │
           ├──────────────────────┐
           │                      │
           ▼                      │
    ┌──────────────┐              │
    │  BatchNorm2d │              │
    └──────┬───────┘              │
           │                      │
           ▼                      │
    ┌──────────────┐              │
    │     LKA      │              │
    └──────┬───────┘              │
           │                      │
           ▼                      │
         (+) ◄────────────────────┘  Residual Connection
           │
           ├──────────────────────┐
           │                      │
           ▼                      │
    ┌──────────────┐              │
    │  BatchNorm2d │              │
    └──────┬───────┘              │
           │                      │
           ▼                      │
    ┌──────────────┐              │
    │ FeedForward  │              │
    └──────┬───────┘              │
           │                      │
           ▼                      │
         (+) ◄────────────────────┘  Residual Connection
           │
           ▼
        Output
```

**Design Rationale:**
- **Pre-normalization:** Stabilizes training, allows learning identity mapping
- **Residual Connections:** Prevents vanishing gradients, enables deeper networks
- **BatchNorm over LayerNorm:** More efficient for CNNs, works well with batch processing

---

### 4.5 Progressive Upsampling

**Purpose:** Increase spatial resolution by 4× using efficient sub-pixel convolution.

```python
self.upsample = nn.Sequential(
    nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1),   # 32 → 128 channels
    nn.PixelShuffle(2),                                             # 2× spatial upscale
    nn.Conv2d(dim, dim * 4, kernel_size=3, stride=1, padding=1),   # 32 → 128 channels
    nn.PixelShuffle(2)                                              # 2× spatial upscale
)
```

#### PixelShuffle Operation

PixelShuffle rearranges elements from channel dimension to spatial dimensions:

```
Before PixelShuffle(r=2):
  Tensor shape: (B, C×r², H, W) = (B, 128, 128, 80)

After PixelShuffle(r=2):
  Tensor shape: (B, C, H×r, W×r) = (B, 32, 256, 160)

Mathematical formulation:
  Output[b, c, h×r+i, w×r+j] = Input[b, c×r²+i×r+j, h, w]
  where i, j ∈ {0, 1, ..., r-1}
```

#### Upsampling Pipeline

```
Input: (B, 32, 128, 80)
           │
           ▼
    Conv2d (32→128)  →  (B, 128, 128, 80)
           │
           ▼
    PixelShuffle(2)  →  (B, 32, 256, 160)   [2× upscale]
           │
           ▼
    Conv2d (32→128)  →  (B, 128, 256, 160)
           │
           ▼
    PixelShuffle(2)  →  (B, 32, 512, 320)   [2× upscale]
           │
           ▼
Output: (B, 32, 512, 320)                    [Total: 4× upscale]
```

#### Why PixelShuffle over Transpose Convolution?

| Aspect | PixelShuffle | Transpose Convolution |
|--------|--------------|----------------------|
| Artifacts | No checkerboard | Prone to checkerboard |
| Efficiency | Higher | Lower |
| Learning | Learnable patterns | Direct upsampling |
| Memory | Lower | Higher |

---

### 4.6 Output Reconstruction

**Purpose:** Convert feature maps back to RGB image space.

```python
self.output = nn.Conv2d(dim, out_channels=3, kernel_size=3, stride=1, padding=1)
```

**Specifications:**
- Input: 32-channel feature map
- Output: 3-channel RGB image
- No activation (regression task)

---

## 5. Data Flow

### Complete Forward Pass

```python
def forward(self, x):
    # Input: (B, 3, 128, 80)
    
    x = self.shallow_feat(x)    # (B, 3, 128, 80) → (B, 32, 128, 80)
    
    x = self.encoder(x)          # (B, 32, 128, 80) → (B, 32, 128, 80)
                                 # [3× TransformerBlockLite]
    
    x = self.upsample(x)         # (B, 32, 128, 80) → (B, 32, 512, 320)
                                 # [4× via dual PixelShuffle]
    
    return self.output(x)        # (B, 32, 512, 320) → (B, 3, 512, 320)
```

### Tensor Shape Progression

| Stage | Operation | Output Shape |
|-------|-----------|--------------|
| Input | - | (B, 3, 128, 80) |
| Shallow Features | Conv2d | (B, 32, 128, 80) |
| Block 1 | TransformerBlockLite | (B, 32, 128, 80) |
| Block 2 | TransformerBlockLite | (B, 32, 128, 80) |
| Block 3 | TransformerBlockLite | (B, 32, 128, 80) |
| Upsample Stage 1 | Conv + PixelShuffle(2) | (B, 32, 256, 160) |
| Upsample Stage 2 | Conv + PixelShuffle(2) | (B, 32, 512, 320) |
| Output | Conv2d | (B, 3, 512, 320) |

---

## 6. Training Configuration

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 2 | Memory constraint (high-resolution outputs) |
| Learning Rate | 2×10⁻⁴ | Standard for Adam with image restoration |
| Epochs | 30 | Sufficient for convergence |
| Optimizer | Adam | Adaptive learning rates |
| Loss Function | L1 Loss (MAE) | Produces sharper images than MSE |

### Loss Function

```python
criterion = nn.L1Loss()
```

**Why L1 over MSE?**
- L1 produces **sharper** reconstructions
- MSE tends to produce **blurry** outputs (penalizes large errors more, leading to averaged predictions)
- L1 is more robust to outliers

### Training Loop

```python
def train_model(model, train_loader, val_loader, device, epochs=30):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for low, high in tqdm(train_loader):
            low, high = low.to(device), high.to(device)
            
            # Forward pass
            output = model(low)
            loss = criterion(output, high)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # Validation
        model.eval()
        psnrs = []
        with torch.no_grad():
            for low, high in val_loader:
                low, high = low.to(device), high.to(device)
                output = model(low)
                psnrs.append(calculate_psnr(output, high))
        
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}, PSNR: {np.mean(psnrs):.2f} dB")
```

### Data Augmentation

```python
transform_lr = transforms.Compose([
    transforms.Resize((128, 80)),
    transforms.ToTensor()
])

transform_hr = transforms.Compose([
    transforms.Resize((512, 320)),
    transforms.ToTensor()
])
```

---

## 7. Implementation Code

### Complete Model Definition

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """Feedforward network with expansion and projection."""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class LKA(nn.Module):
    """Large Kernel Attention module."""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        u = x
        attn = self.conv1(x)
        attn = self.conv2(attn)
        attn = self.conv3(attn)
        return u * attn


class TransformerBlockLite(nn.Module):
    """Lightweight transformer block with LKA and FFN."""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = LKA(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = FeedForward(dim, ff_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RestormerLite(nn.Module):
    """
    Restormer-Lite: Lightweight image restoration model.
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB)
        dim (int): Base channel dimension (default: 32)
        num_blocks (int): Number of transformer blocks (default: 3)
        ff_dim (int): Feedforward hidden dimension (default: 64)
    """
    def __init__(self, in_channels=3, dim=32, num_blocks=3, ff_dim=64):
        super().__init__()
        
        # Shallow feature extraction
        self.shallow_feat = nn.Conv2d(in_channels, dim, 3, 1, 1)
        
        # Transformer encoder
        self.encoder = nn.Sequential(
            *[TransformerBlockLite(dim, ff_dim) for _ in range(num_blocks)]
        )
        
        # Progressive upsampling (4×)
        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(dim, dim * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        
        # Output reconstruction
        self.output = nn.Conv2d(dim, 3, 3, 1, 1)

    def forward(self, x):
        x = self.shallow_feat(x)
        x = self.encoder(x)
        x = self.upsample(x)
        return self.output(x)
```

### PSNR Calculation

```python
from math import log10

def calculate_psnr(pred, target):
    """
    Calculate Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted image tensor
        target: Ground truth image tensor
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(pred, target)
    return 10 * log10(1 / mse.item())
```

---

## 8. Performance Metrics

### Training Progress

| Epoch | Train Loss | Validation PSNR |
|-------|------------|-----------------|
| 1 | 0.0277 | 35.40 dB |
| 5 | 0.0132 | 37.02 dB |
| 10 | 0.0118 | 37.49 dB |
| 15 | 0.0113 | 37.65 dB |
| 20 | 0.0111 | 37.77 dB |
| 25 | 0.0111 | 37.76 dB |
| 30 | 0.0109 | 37.87 dB |

### Final Results

| Metric | Value |
|--------|-------|
| Best Validation PSNR | **37.92 dB** |
| Final Training Loss | 0.0109 |
| Total Training Time | ~11 minutes (30 epochs) |
| GPU | Tesla T4 |

### Model Statistics

| Property | Value |
|----------|-------|
| Total Parameters | ~150K |
| Model Size | ~600 KB |
| Inference Time | ~80 ms per image |
| Memory Footprint | ~1.5 GB (training) |

---

## 9. Comparison with Other Approaches

### Restormer-Lite vs Enhanced SwinIR

| Component | Restormer-Lite | Enhanced SwinIR |
|-----------|----------------|-----------------|
| **Attention Mechanism** | Large Kernel Attention (LKA) | Channel Attention (SE-style) |
| **Core Block** | TransformerBlockLite (LKA + FFN) | ResidualBlock (Conv + BatchNorm) |
| **Normalization** | BatchNorm2d | BatchNorm2d |
| **Upsampling** | PixelShuffle × 2 | UpSampleBlock2x × 2 |
| **Loss Function** | L1 Loss | MSE Loss |
| **Number of Blocks** | 3 | 8 |
| **Best PSNR** | 37.92 dB | 37.99 dB |
| **Training Epochs** | 30 | 20 |
| **Batch Size** | 2 | 4 |

### Architectural Philosophy

**Restormer-Lite:**
- Focus on **spatial attention** via large receptive fields
- Fewer blocks with more powerful attention
- Better for capturing global context

**Enhanced SwinIR:**
- Focus on **channel attention** via SE-style mechanism
- More blocks with simpler operations
- Better for channel-wise feature recalibration

---

## 10. References

1. **Restormer:** Zamir, S.W., et al. "Restormer: Efficient Transformer for High-Resolution Image Restoration." CVPR 2022.

2. **SwinIR:** Liang, J., et al. "SwinIR: Image Restoration Using Swin Transformer." ICCV 2021.

3. **PixelShuffle:** Shi, W., et al. "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network." CVPR 2016.

4. **Large Kernel Attention:** Guo, M.H., et al. "Visual Attention Network." arXiv 2022.

---

## Appendix: Tools & Technologies

- **Framework:** PyTorch
- **Libraries:** torchvision, PIL, NumPy, tqdm
- **Training Platform:** Kaggle (Tesla T4 GPU)
- **Development:** Python 3.10

---

**Document End**

*For questions or collaborations, contact: 21f1003346@ds.study.iitm.ac.in*
