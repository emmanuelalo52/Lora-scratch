#Build LoRa from scratch
#Dependencies
#lora_a,lora_b, embedding of both low rank adaptors
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,List

class LoraLayer(nn.Module):
    def __init__(self, rank, alpha, base_layer, merge_weights=True, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.base_layer = base_layer
        self.merge_weights = merge_weights

        # Initialize scaling
        self.scaling = self.alpha / self.rank

        # Get the shape of the base layer's weight matrix
        if isinstance(base_layer, nn.Linear):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        elif isinstance(base_layer, nn.Embedding):
            self.in_features = base_layer.num_embeddings
            self.out_features = base_layer.embedding_dim
        else:
            raise ValueError("Unsupported base layer type. Only nn.Linear and nn.Embedding are supported.")

        # Initialize LoRA matrices
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros((self.in_features, rank)))  # LoRA A matrix
            self.lora_B = nn.Parameter(torch.zeros((rank, self.out_features)))  # LoRA B matrix
        else:
            self.lora_A = None
            self.lora_B = None

        # Track whether weights are merged
        self.merged = False

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        if self.rank > 0:
            nn.init.normal_(self.lora_A, mean=0, std=0.02)
            nn.init.zeros_(self.lora_B)

    def merge_weights(self):
        if self.merge_weights and not self.merged and self.rank > 0:
            # LoRA B * LoRA A (Matrix multiplication)
            lora_adaptor = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data += lora_adaptor
            self.merged = True

    def unmerge_weights(self):
        if self.merge_weights and self.merged and self.rank > 0:
            # LoRA B * LoRA A (Matrix multiplication)
            lora_adaptor = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.data -= lora_adaptor
            self.merged = False

    def forward(self, x):
        # Forward pass through the base layer
        result = self.base_layer(x)

        # Add LoRA adaptation if rank > 0 and weights are not merged
        if self.rank > 0 and not self.merged:
            if isinstance(self.base_layer, nn.Linear):
                lora_adaptation = (x @ self.lora_A) @ self.lora_B  # LoRA adaptation for Linear
            elif isinstance(self.base_layer, nn.Embedding):
                lora_adaptation = F.embedding(x, self.lora_A.transpose(0, 1)) @ self.lora_B  # LoRA adaptation for Embedding
            else:
                raise ValueError("Unsupported base layer type.")

            # Apply dropout to the LoRA adaptation
            if self.dropout > 0:
                lora_adaptation = F.dropout(lora_adaptation, p=self.dropout, training=self.training)

            # Scale and add to result
            result += lora_adaptation * self.scaling

        return result
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, rank, alpha=1, merge_weights=True,
                 padding_idx=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.rank = rank
        self.merge_weights = merge_weights
        self.alpha = alpha
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Initialize main embedding weights
        self.weight = nn.Parameter(torch.empty((vocab_size, embed_dim)))

        # Initialize LoRA adapters
        assert rank > 0
        self.lora_A = nn.Parameter(torch.zeros((vocab_size, rank)))  # LoRA A matrix
        self.lora_B = nn.Parameter(torch.zeros((rank, embed_dim)))  # LoRA B matrix

        # Scaling factor
        self.scaling = self.alpha / self.rank

        # Track whether weights are merged
        self.merged = False

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=1 / math.sqrt(self.vocab_size))
        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def merge_weights(self):
        if self.merge_weights and not self.merged:
            if self.rank > 0:
                # LoRA B * LoRA A (Matrix multiplication)
                lora_adaptor = (self.lora_B @ self.lora_A) * self.scaling
                self.weight.data += lora_adaptor
                self.merged = True

    def unmerge_weights(self):
        if self.merge_weights and self.merged:
            if self.rank > 0:
                # LoRA B * LoRA A (Matrix multiplication)
                lora_adaptor = (self.lora_B @ self.lora_A) * self.scaling
                self.weight.data -= lora_adaptor
                self.merged = False

    def forward(self, x):
        if self.rank > 0 and not self.merged:
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"Input must be a torch.Tensor, got {type(x)}")
            #nn.Embedding.forward(self, x)
            x_emb = self.weight[x]
            if self.rank > 0 and not self.merged:
                after_A = x @ self.lora_A  # Direct matrix multiplication
                x_emb += (after_A @ self.lora_B) * self.scaling
            return x_emb
class LinearWithLoRA(nn.Module):
    def __init__(self, fan_in, fan_out, rank, alpha=1, bias=True, merge_weights=True):
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights
        self.scaling = alpha / rank

        # Initialize main weight matrix
        self.weight = nn.Parameter(torch.empty((fan_in, fan_out)))

        # Initialize bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(fan_out))
        else:
            self.bias = None

        # Initialize LoRA matrices
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros((fan_in, rank)))  # LoRA A matrix
            self.lora_B = nn.Parameter(torch.zeros((rank, fan_out)))  # LoRA B matrix
        else:
            self.lora_A = None
            self.lora_B = None

        # Track whether weights are merged
        self.merged = False

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize main weight matrix
        nn.init.normal_(self.weight, mean=0, std=1 / math.sqrt(self.fan_in))

        # Initialize LoRA matrices
        if self.rank > 0:
            nn.init.normal_(self.lora_A, mean=0, std=0.02)
            nn.init.zeros_(self.lora_B)

    def merge_weights(self):
        if self.merge_weights and not self.merged and self.rank > 0:
            # LoRA B * LoRA A (Matrix multiplication)
            lora_adaptor = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data += lora_adaptor
            self.merged = True

    def unmerge_weights(self):
        if self.merge_weights and self.merged and self.rank > 0:
            # LoRA B * LoRA A (Matrix multiplication)
            lora_adaptor = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data -= lora_adaptor
            self.merged = False

    def forward(self, x):
        # Standard linear transformation
        result = x @ self.weight

        # Add LoRA adaptation if rank > 0 and weights are not merged
        if self.rank > 0 and not self.merged:
            lora_adaptation = (x @ self.lora_A) @ self.lora_B  # LoRA adaptation
            result += lora_adaptation * self.scaling  # Scale and add to result

        # Add bias
        if self.bias is not None:
            result += self.bias

        return result