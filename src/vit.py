import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

import math

from dataclasses import dataclass
from datasets import load_dataset


torch.manual_seed(42)
torch.cuda.manual_seed(42)


@dataclass
class ViTConfig:
    n_layer: int = 12
    d_size: int = 192
    n_head: int = 3
    n_class: int = 1000
    patch_size: int = 16
    image_size: int = 224
    num_channels: int = 3


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ViTPatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.projection = nn.Conv2d(self.config.num_channels, self.config.d_size, 
                                    kernel_size=self.config.patch_size, stride=self.config.patch_size)

    def forward(self, x):
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class ViTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        num_patches = (config.image_size // config.patch_size) ** 2

        self.config = config
        self.cls_token = nn.Parameter(torch.rand(1, 1, self.config.d_size))
        self.position_embeddings = nn.Parameter(torch.rand(1, num_patches+1, self.config.d_size))
        self.patch_embeddings = ViTPatchEmbedding(config)


    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)                  # add cls_token
        # print(torch.cat((cls_token,self.patch_embeddings(x)), dim=1).shape)
        # print(cls_token.shape, self.patch_embeddings(x).shape)
        embeddings = torch.cat((cls_token, self.patch_embeddings(x)), dim=1) + self.position_embeddings  # make positional embeddings

        return embeddings


class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.query = nn.Linear(self.config.d_size, self.config.d_size, bias=True)
        self.key = nn.Linear(self.config.d_size, self.config.d_size, bias=True)
        self.value = nn.Linear(self.config.d_size, self.config.d_size, bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.config.n_head, self.config.d_size // self.config.n_head)
        x = x.view(new_x_shape)       # (B, N_embd, N_head, D_head)

        return x.permute(0, 2, 1, 3)  # (B, N_head, N_embd, D_head)

    def forward(self, x):
        key_layer = self.transpose_for_scores(self.key(x))
        query_layer = self.transpose_for_scores(self.query(x))
        value_layer = self.transpose_for_scores(self.value(x))

        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores = attention_scores / math.sqrt(self.config.d_size // self.config.n_head)
        # attention_probs = F.softmax(attention_scores, dim=-1)
        # context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, is_causal=True)  # utilize FlashAttention ### 3

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.config.d_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class ViTOutput(nn.Module):
    def __init__(self, config, type='self'):
        super().__init__()

        self.config = config
        self.dense = nn.Linear(self.config.d_size, self.config.d_size) if type == 'self' else nn.Linear(self.config.d_size * 4, self.config.d_size)

    def forward(self, x):
        return self.dense(x)


class ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.attention = nn.ModuleDict(dict(
            attention = ViTSelfAttention(self.config),
            output = ViTOutput(self.config, type='self'),
        ))
        self.intermediate = nn.ModuleDict(dict(
            dense = nn.Linear(self.config.d_size, self.config.d_size * 4),
            act_fn = nn.GELU()
        ))
        # self.output = nn.Linear(self.config.d_size * 4, self.config.d_size)
        self.output = ViTOutput(self.config, type="output")
        self.layernorm_before = nn.LayerNorm(self.config.d_size)
        self.layernorm_after = nn.LayerNorm(self.config.d_size)

    def forward(self, x):
        x = self.attention.output(self.attention.attention(self.layernorm_before(x))) + x                  # layer norm before self attention + skip connection
        x = self.output(self.intermediate.act_fn(self.intermediate.dense((self.layernorm_after(x))))) + x   # layer norm after self attention + skip connection
                 
        return x


class ViTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layer = nn.ModuleList([ViTLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        
        return x


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.vit = nn.ModuleDict(dict(
            embeddings = ViTEmbeddings(self.config),
            encoder = ViTEncoder(self.config),
            layernorm = nn.LayerNorm(self.config.d_size),
        ))

        self.classifier = nn.Linear(self.config.d_size, self.config.n_class, bias=True)

    def forward(self, x):
        # x = self.vit(x)
        x = self.vit.embeddings(x)
        # print("after embd", x.shape)
        x = self.vit.encoder(x)
        # print("x after encoder is ", x.shape)
        x = x[:, 0, :]
        # print("cls token extracted ", x.shape)
        x = self.vit.layernorm(x)
        x = self.classifier(x)

        return x

    @staticmethod
    def get_id2label(model_type):
        from transformers import ViTForImageClassification

        model_pretrained = ViTForImageClassification.from_pretrained(model_type)

        return model_pretrained.config.id2label

    @staticmethod
    def from_pretrained(model_type, n_class=None):
        assert model_type in {'facebook/deit-tiny-patch16-224'} # single model for now

        from transformers import ViTForImageClassification
        print("loading weights from", model_type)

        config_args = {
            'facebook/deit-tiny-patch16-224': dict(n_layer=12, n_head=3, d_size=192, n_class=1000),
        }[model_type]

        if n_class is not None:
            config_args["n_class"] = n_class
        config_args["patch_size"] = 16
        config_args["image_size"] = 224
        config_args["num_channels"] = 3

        config = ViTConfig(**config_args)
        model = ViT(config)
        sd = model.state_dict()

        model_pretrained = ViTForImageClassification.from_pretrained(model_type)
        sd_pretrained = model_pretrained.state_dict()
        sd_keys_pretrained = sd_pretrained.keys()

        """
        sd_keys = sd.keys()

        print(set(sd_keys) == set(model_pretrained.state_dict().keys())) # check if names of layers match

        for k, v in sd.items():                           # check shapes of layers to match
            if v.shape != sd_pretrained[k].shape:
                print("Shapes do not match at", k)
        """

        # copy weights
        for k in sd_keys_pretrained:
            if k == 'classifier.weight' or k == 'classifier.bias':
                print(f"initializing {k} from scratch")
            else:
                assert sd[k].shape == sd_pretrained[k].shape, f'at {k}'
                with torch.no_grad():
                    sd[k].copy_(sd_pretrained[k])

        print("loading finished!")

        return model

    

if __name__ == "__main__":
    
    model = ViT(ViTConfig()).from_pretrained('facebook/deit-tiny-patch16-224').eval().to(device)

    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = pil_to_tensor(dataset["test"]["image"][0].resize((224, 224))).unsqueeze(0).to(device)
    image = image.type(torch.float32) / 255.

    with torch.no_grad():
        out = model(image)
        print(out.shape)

    predicted_label = out.argmax(-1).item()

    id2label = model.get_id2label('facebook/deit-tiny-patch16-224')
    print(id2label[predicted_label])
