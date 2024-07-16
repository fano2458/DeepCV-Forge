import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass


@dataclass
class ViTConfig:
    n_layer: int = 12
    n_embd: int = 192
    n_head: int = 12
    n_class: int = 1000


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ViTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def forward(self):
        pass


class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super.__init__()

        self.config = config

    def forward(self):
        pass


class ViTLayer(nn.Module):
    def __init__(self, config):
        super.__init__()

        self.config = config
        self.attention = ViTSelfAttention(self.config)
        self.intermediate = nn.ModuleDict(dict(
            dense = nn.Linear(self.config.n_embd, self.config.n_embd * 4),
            act_fn = nn.GELU()                              # TODO check
        ))
        self.output = nn.Linear(self.config.n_embd * 4, self.config.n_embd)
        self.layernorm_before = nn.LayerNorm(self.config.n_embd)
        self.layernorm_after = nn.LayerNorm(self.config.n_embd)

    def forward(self):
        pass


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super.__init__()

        self.config = config
        self.layer = nn.ModuleList([ViTLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self):
        pass


class ViT(nn.Module):
    def __init__(self, config):
        super.__init__()

        self.config = config

        self.vit = nn.ModuleDict(dict(
            embeddings = ViTEmbeddings(self.config),
            encoder = ViTEncoder(self.config),
            layernorm = nn.LayerNorm(self.config.n_embd),
        ))

        self.classifier = nn.Linear(self.config.n_embd, self.config.n_class, bias=True)

    def forward(self):
        pass

    @staticmethod
    def from_pretrained(self, x):
        pass

    