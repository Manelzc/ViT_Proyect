import torch
import torch.nn as nn
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 16, emb_size = 768, img_size = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size), # Capa convolucional 
            Rearrange('b e (h) (w) -> b (h w) e'),                                       # einops
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1) # Concatenar cls_token al input
        x += self.positions # Agregar position embedding
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size = 768,
                 drop_p: float = 0.,
                 forward_expansion = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])



class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size = 768, n_classes = 1000): # CIFAR y ImageNet tienen 1000 clases, normalmente se define asi
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size = 768, num_heads = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # Fusionar Values, Queries y Keys en una matriz:
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # Separar Values, Queries y Keys en num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # Ultimo eje
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values) # Tercer eje
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out




class ViT(nn.Sequential):
    # ViT-Base: `depth=12, emb_size=768, mlp_size=3072, n_heads=12`
    # Estos son los parámetros originales mencionados en el paper para el ViT-Base
    def __init__(self,
                in_channels = 3,
                patch_size = 16,
                emb_size = 768,
                img_size = 224,
                depth = 12,
                n_classes = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
if __name__=="__main__":
    x = torch.randn((16, 3, 224, 224))
    patches_embedded = PatchEmbedding()(x)
    print(TransformerEncoderBlock()(patches_embedded).shape)
