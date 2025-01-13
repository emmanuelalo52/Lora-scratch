#Build LoRa from scratch
#Dependencies
#lora_a,lora_b, embedding of both low rank adaptors
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional,List

class Embedding(nn.Module):
    def __init__(self,vocab_size,embed_dim,rank,alpha=1,merge_weights=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.rank = rank
        self.merge_weights = merge_weights
        self.alpha = alpha
        #initialize a weight at 0
        self.weight = nn.Parameter(torch.empty((vocab_size,embed_dim)))
        #initialize the adaptors
        assert rank > 0
        self.lora_A = nn.Parameter(self.weight.new_zeros((vocab_size,rank)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((rank, embed_dim)))

        self.scaling = self.alpha/self.rank

        self.reset_parameters()
    def reset_parameters(self):
        nn.init.normal_(self.weight,mean=0,std=1/math.sqrt(self.vocab_size))
        nn.init.normal_(self.lora_A,mean=0,std=0.02)
        nn.init.zeros_(self.lora_B)
    def merge_unmerge_weight(self):
        if self.merge_weights and self.merged:
            if self.rank > 0:
                #Lora B * Lora A (Matrix multiplication)
                lora_adaptor = (self.lora_B @ self.lora_A) * self.scaling
                self.weight.data += lora_adaptor
                self.merged = True
        else:
            if self.merge_weights and not self.merged:
                if self.rank > 0:
                    lora_adaptor = (self.lora_B @ self.lora_A) * self.scaling
                    self.weight.data -= lora_adaptor
                    self.merged = False
    def forward(self, x):
        B,T,C = x.shape
        