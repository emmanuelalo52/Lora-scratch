# PyTorch LoRA (Low-Rank Adaptation) Implementation

A comprehensive implementation of Low-Rank Adaptation (LoRA) in PyTorch, featuring both theoretical foundations and practical implementation details. This repository contains the complete implementation of LoRA for neural network fine-tuning, with special focus on Linear and Embedding layers.

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation Details](#implementation-details)
4. [Code Structure](#code-structure)
5. [Usage Guide](#usage-guide)
6. [Advanced Features](#advanced-features)

## Introduction

Low-Rank Adaptation (LoRA) is a technique for efficient fine-tuning of large neural networks. Instead of updating all parameters during fine-tuning, LoRA injects trainable rank decomposition matrices into the original network, significantly reducing the number of trainable parameters while maintaining performance.

## Mathematical Foundation

### Core Concept

From the provided mathematical documentation (`Lora math.pdf`), the key equations and concepts are:

1. **Weight Decomposition**:
   - Original weight update: ΔW ∈ ℝᵈˣᵏ
   - LoRA decomposition: ΔW = BA, where:
     - B ∈ ℝᵈˣʳ
     - A ∈ ℝʳˣᵏ
     - r is the rank (typically r ≪ min(d,k))

2. **Scaling Factor**:
   - Final weight modification: W + α(BA)/r
   - Where α is a scaling factor and r is the rank

3. **Training Dynamics**:
   ```
   W = W₀ + ΔW
   ΔW = α(BA)/r
   ```

## Implementation Details

The implementation consists of three main classes, each serving a specific purpose in the LoRA architecture.

### 1. Base LoRA Layer (`LoraLayer`)

```python
class LoraLayer(nn.Module):
    def __init__(self, rank, alpha, base_layer, merge_weights=True, dropout=0.1):
        # ... initialization code ...
```

Key features:
- Universal wrapper for both Linear and Embedding layers
- Supports dynamic weight merging/unmerging
- Implements dropout for regularization
- Handles both forward and backward passes efficiently

### 2. LoRA Embedding Implementation

```python
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, rank, alpha=1, merge_weights=True,
                 padding_idx=None, max_norm=None, norm_type=2, 
                 scale_grad_by_freq=False, sparse=False):
        # ... initialization code ...
```

Special features:
- Handles vocabulary embeddings with LoRA adaptation
- Supports padding indices
- Implements weight normalization options
- Provides sparse gradient options

### 3. Linear Layer with LoRA

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, fan_in, fan_out, rank, alpha=1, bias=True, merge_weights=True):
        # ... initialization code ...
```

Implementation details:
- Direct implementation for linear transformations
- Optional bias terms
- Efficient matrix multiplication ordering
- Support for weight merging

## Code Structure

### Core Components Breakdown

1. **Initialization Logic**:
```python
# LoRA matrices initialization
self.lora_A = nn.Parameter(torch.zeros((self.in_features, rank)))
self.lora_B = nn.Parameter(torch.zeros((rank, self.out_features)))

# Scaling factor
self.scaling = self.alpha / self.rank
```

2. **Parameter Reset**:
```python
def reset_parameters(self):
    nn.init.normal_(self.weight, mean=0, std=1/math.sqrt(self.vocab_size))
    nn.init.normal_(self.lora_A, mean=0, std=0.02)
    nn.init.zeros_(self.lora_B)
```

3. **Weight Management**:
```python
def merge_weights(self):
    if self.merge_weights and not self.merged:
        lora_adaptor = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.data += lora_adaptor
        self.merged = True
```

### Forward Pass Implementation

Different forward passes for each layer type:

1. **Embedding Forward Pass**:
```python
def forward(self, x):
    if self.rank > 0 and not self.merged:
        x_emb = self.weight[x]
        after_A = x @ self.lora_A
        x_emb += (after_A @ self.lora_B) * self.scaling
        return x_emb
```

2. **Linear Forward Pass**:
```python
def forward(self, x):
    result = x @ self.weight
    if self.rank > 0 and not self.merged:
        lora_adaptation = (x @ self.lora_A) @ self.lora_B
        result += lora_adaptation * self.scaling
    if self.bias is not None:
        result += self.bias
    return result
```

## Usage Guide

### Basic Setup

```python
# Create a LoRA-adapted embedding layer
embedding = Embedding(
    vocab_size=50000,    # Vocabulary size
    embed_dim=768,       # Embedding dimension
    rank=16,             # LoRA rank
    alpha=32,            # Scaling factor
    merge_weights=True   # Enable weight merging
)

# Create a LoRA-adapted linear layer
linear = LinearWithLoRA(
    fan_in=768,          # Input dimension
    fan_out=768,         # Output dimension
    rank=16,             # LoRA rank
    alpha=32,            # Scaling factor
    bias=True            # Use bias
)
```

### Training Loop Example

```python
# Initialize optimizer
optimizer = torch.optim.AdamW([
    {'params': embedding.parameters()},
    {'params': linear.parameters()}
], lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        embeds = embedding(batch['input_ids'])
        output = linear(embeds)
        
        # Calculate loss and backward pass
        loss = criterion(output, batch['labels'])
        loss.backward()
        
        optimizer.step()
```

## Advanced Features

### 1. Weight Management
- Dynamic merging/unmerging for inference
- Memory-efficient training
- Support for quantization

### 2. Regularization
- Dropout implementation
- Weight normalization
- Gradient scaling

### 3. Architecture Support
- Compatible with transformer architectures
- Extensible to other layer types
- Support for complex neural networks

## Debugging and Common Issues

1. Memory Management:
   - Use weight merging for inference
   - Implement gradient checkpointing if needed
   - Monitor memory usage during training

2. Training Stability:
   - Start with smaller rank values
   - Adjust alpha based on task
   - Monitor gradient norms

3. Performance Optimization:
   - Use efficient matrix multiplication ordering
   - Implement batch processing
   - Enable weight merging during inference

## Contributing

Contributions are welcome! Areas of interest:
1. Additional layer type support
2. Performance optimizations
3. Extended documentation
4. Test coverage
5. Training examples

## License

This project is licensed under the MIT License.

