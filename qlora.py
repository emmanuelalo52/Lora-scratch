import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1, bias=True, merge_weights=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights
        self.scaling = alpha / rank
        self.merged = False

        # Quantized base layer
        self.base_layer = bnb.nn.Linear4bit(
            in_features, 
            out_features, 
            bias=bias
        )
        
        # LoRA adapters
        self.lora_A = nn.Parameter(torch.zeros((in_features, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_features)))
        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.base_layer(x)
        if self.rank > 0 and not self.merged:
            lora_adaptation = (x @ self.lora_A) @ self.lora_B
            result += lora_adaptation * self.scaling
        return result

class QLoRAEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, rank, alpha=1, merge_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights
        self.scaling = alpha / rank
        self.merged = False

        # Using quantized linear layer for embeddings
        self.base_layer = bnb.nn.Linear4bit(
            vocab_size,
            embed_dim,
            bias=False
        )

        # LoRA adapters
        self.lora_A = nn.Parameter(torch.zeros((vocab_size, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, embed_dim)))
        nn.init.normal_(self.lora_A, mean=0, std=0.02)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Convert indices to one-hot vectors
        one_hot = F.one_hot(x, num_classes=self.vocab_size).float()
        # Get embeddings through linear layer
        x_emb = self.base_layer(one_hot)
        
        if self.rank > 0 and not self.merged:
            # Use one-hot vectors for LoRA adaptation
            lora_adaptation = (one_hot @ self.lora_A) @ self.lora_B
            x_emb += lora_adaptation * self.scaling
        return x_emb

