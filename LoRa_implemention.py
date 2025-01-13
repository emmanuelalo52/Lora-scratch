#Build LoRa from scratch
#Dependencies
#lora_a,lora_b, embedding of both low rank adaptors
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,List

class LoraLayer(nn.Module):
    def __init__(self,rank,alpha,lora_weight,bias=False,dropout=0.1):
        super().__init__()
        self.rank = rank
        self.dropout = dropout
        self.alpha = alpha
        self.lora_weight = lora_weight
        self.bias = bias
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
        self.lora_A = nn.Parameter(self.weight.new_zeros((vocab_size, rank)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((rank, embed_dim)))

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
            x_emb = self.weight[x]
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            x_emb += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return x_emb
        else:
            return nn.Embedding.forward(self, x)     
class LinearWithLoRA(nn.Module):
    def __init__(self, fan_in, fan_out, rank=8, alpha=1.0, bias=True, merge_weights=True):
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights
        self.scaling = alpha / rank

        #initialize weight
        self.weight = nn.Parameter(torch.empty((fan_in,fan_out)))

        #initialize bias too
        if bias is not None:
            self.bias = nn.Parameter(torch.zeros(fan_out))
        else:
            self.bias = None
        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.zeros((fan_in,rank)))
            self.lora_B = nn.Parameter(torch.zeros((rank,fan_out)))
        else:
            self.lora_A = None
            self.lora_B = None
        #track merging
        self.merged = False

        self.reset_parameters()
    def reset_parameters(self):
        # Initialize main weight matrix
        nn.init.normal_(self.weight, mean=0, std=1 / math.sqrt(self.in_features))

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
    def forward(self,x):
        result = x @ self.weight
        if self.rank > 0 and not self.merged:
            lora_adaptator = (x @ self.lora_A) @ self.lora_B  # LoRA adaptator
            result += lora_adaptator * self.scaling  # Scale and add to result
        if self.bias is not None:
            result += self.bias
        return result