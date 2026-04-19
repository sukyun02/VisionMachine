"""
DHVT (Dual-Head Vision Transformer) — timm-free implementation.
Based on "Bridging the Gap Between Vision Transformers and Convolutional Neural Networks
on Small Datasets" (NeurIPS 2022).

Original code: https://github.com/ArieSeirwormo/DHVT
Modifications: removed timm dependency, implemented utility functions inline.
"""
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities (replacing timm.models.layers)
# ---------------------------------------------------------------------------

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization (approximation)."""
    with torch.no_grad():
        nn.init.normal_(tensor, mean=mean, std=std)
        tensor.clamp_(mean + a * std, mean + b * std)
    return tensor


def lecun_normal_(tensor):
    fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
    std = math.sqrt(1.0 / fan_in)
    trunc_normal_(tensor, std=std)
    return tensor


def named_apply(fn, module, name='', depth_first=True, include_root=True):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name,
                    depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


# ---------------------------------------------------------------------------
# Drop Path
# ---------------------------------------------------------------------------

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# MLP / DAFF
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DAFF(nn.Module):
    """Depth-Aware Feed-Forward with depthwise convolutions and SE-style gating."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., kernel_size=3, with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.conv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size, 1,
                               (kernel_size - 1) // 2, groups=hidden_features)
        self.conv3 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.act = act_layer()
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.bn2 = nn.BatchNorm2d(hidden_features)
        self.bn3 = nn.BatchNorm2d(out_features)
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Linear(in_features, in_features // 4)
        self.excitation = nn.Linear(in_features // 4, in_features)

    def forward(self, x):
        B, N, C = x.size()
        cls_token, tokens = torch.split(x, [1, N - 1], dim=1)
        x = tokens.reshape(B, int(math.sqrt(N - 1)), int(math.sqrt(N - 1)), C).permute(0, 3, 1, 2)

        x = self.act(self.bn1(self.conv1(x)))
        shortcut = x
        x = self.act(self.bn2(self.conv2(x)))
        x = shortcut + x
        x = self.bn3(self.conv3(x))

        weight = self.squeeze(x).flatten(1).reshape(B, 1, C)
        weight = self.excitation(self.act(self.compress(weight)))
        cls_token = cls_token * weight

        tokens = x.flatten(2).permute(0, 2, 1)
        return torch.cat((cls_token, tokens), dim=1)


# ---------------------------------------------------------------------------
# Patch Embeddings
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Standard non-overlapping patch embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_planes),
    )


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, dim, 1, 1]))
        self.beta = nn.Parameter(torch.zeros([1, dim, 1, 1]))

    def forward(self, x):
        return x * self.alpha + self.beta


class ConvPatchEmbed(nn.Module):
    """SOPE: CNN-based patch embedding replacing the single linear projection."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, init_values=1e-2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        if patch_size[0] == 16:
            self.proj = nn.Sequential(
                _conv3x3(3, embed_dim // 8, 2), nn.GELU(),
                _conv3x3(embed_dim // 8, embed_dim // 4, 2), nn.GELU(),
                _conv3x3(embed_dim // 4, embed_dim // 2, 2), nn.GELU(),
                _conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 4:
            self.proj = nn.Sequential(
                _conv3x3(3, embed_dim // 2, 2), nn.GELU(),
                _conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 2:
            self.proj = nn.Sequential(
                _conv3x3(3, embed_dim, 2), nn.GELU(),
            )
        else:
            raise ValueError("ConvPatchEmbed: patch_size must be in [2, 4, 16]")

        self.pre_affine = Affine(3)
        self.post_affine = Affine(embed_dim)

    def forward(self, x):
        x = self.pre_affine(x)
        x = self.proj(x)
        x = self.post_affine(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class HI_Attention(nn.Module):
    """Head-Independent Multi-Head Self-Attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.GELU()
        self.ht_proj = nn.Linear(dim // num_heads, dim, bias=True)
        self.ht_norm = nn.LayerNorm(dim // num_heads)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_heads, dim))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B, N, C = x.shape

        # head token
        head_pos = self.pos_embed.expand(B, -1, -1)
        x_ = x.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x_ = x_.mean(dim=2)
        x_ = self.ht_proj(x_).reshape(B, -1, self.num_heads, C // self.num_heads)
        x_ = self.act(self.ht_norm(x_)).flatten(2)
        x_ = x_ + head_pos
        x = torch.cat([x, x_], dim=1)

        # normal mhsa
        qkv = self.qkv(x).reshape(B, N + self.num_heads, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N + self.num_heads, C)
        x = self.proj(x)

        # merge head tokens into cls token
        cls, patch, ht = torch.split(x, [1, N - 1, self.num_heads], dim=1)
        cls = cls + torch.mean(ht, dim=1, keepdim=True)
        x = torch.cat([cls, patch], dim=1)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    """Standard multi-head self-attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Standard ViT encoder block."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., qk_scale=None, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DHVT_Block(nn.Module):
    """DHVT encoder block with HI-MHSA and DAFF."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., qk_scale=None, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = HI_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                 attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DAFF(in_features=dim, hidden_features=mlp_hidden_dim,
                        act_layer=act_layer, drop=drop, kernel_size=3)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Vision Transformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """Vision Transformer with optional DHVT blocks (HI-MHSA + DAFF + SOPE)."""

    def __init__(self, img_size=32, patch_size=16, in_chans=3, num_classes=100,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 weight_init='', apply_dhvt=True, num_superclasses=0):
        super().__init__()
        self.img_size = img_size
        self.depth = depth
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.apply_dhvt = apply_dhvt

        # Patch Embedding
        if self.apply_dhvt:
            self.patch_embed = ConvPatchEmbed(img_size=img_size, embed_dim=embed_dim,
                                             patch_size=patch_size)
        else:
            self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size,
                                          in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if not self.apply_dhvt:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        block_cls = DHVT_Block if self.apply_dhvt else Block
        self.blocks = nn.ModuleList([
            block_cls(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                      drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Auxiliary superclass head
        self.num_superclasses = num_superclasses
        self.head_superclass = nn.Linear(self.num_features, num_superclasses) if num_superclasses > 0 else None

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if not self.apply_dhvt:
            trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward_features(self, x):
        B, _, h, w = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(B, -1, -1), x), dim=1)

        if not self.apply_dhvt:
            pos_embed = self.pos_embed
            if x.shape[1] != pos_embed.shape[1]:
                assert h == w
                real_pos = pos_embed[:, self.num_tokens:]
                hw = int(math.sqrt(real_pos.shape[1]))
                true_hw = int(math.sqrt(x.shape[1] - self.num_tokens))
                real_pos = real_pos.transpose(1, 2).reshape(1, self.embed_dim, hw, hw)
                new_pos = F.interpolate(real_pos, size=true_hw, mode='bicubic', align_corners=False)
                new_pos = new_pos.reshape(1, self.embed_dim, -1).transpose(1, 2)
                pos_embed = torch.cat([pos_embed[:, :self.num_tokens], new_pos], dim=1)
            x = self.pos_drop(x + pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        features = self.forward_features(x)
        if self.head_dist is not None:
            x_fine, x_dist = self.head(features[0]), self.head_dist(features[1])
            if self.training and not torch.jit.is_scripting():
                if self.head_superclass is not None:
                    coarse_logits = self.head_superclass(features[0])
                    return x_fine, x_dist, coarse_logits
                return x_fine, x_dist
            else:
                fine_logits = (x_fine + x_dist) / 2
        else:
            fine_logits = self.head(features)

        if self.head_superclass is not None and self.training:
            coarse_logits = self.head_superclass(features)
            return fine_logits, coarse_logits

        return fine_logits


# ---------------------------------------------------------------------------
# Weight initialization
# ---------------------------------------------------------------------------

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0.,
                      jax_impl: bool = False):
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------

def dhvt_tiny_cifar_patch4(num_classes=100, num_superclasses=0, drop_rate=0.,
                           drop_path_rate=0., **kwargs):
    return VisionTransformer(
        img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=4,
        mlp_ratio=4, apply_dhvt=True, num_classes=num_classes,
        num_superclasses=num_superclasses, drop_rate=drop_rate,
        drop_path_rate=drop_path_rate, **kwargs)


def dhvt_small_cifar_patch4(num_classes=100, num_superclasses=0, drop_rate=0.,
                            drop_path_rate=0., **kwargs):
    return VisionTransformer(
        img_size=32, patch_size=4, embed_dim=384, depth=12, num_heads=8,
        mlp_ratio=4, apply_dhvt=True, num_classes=num_classes,
        num_superclasses=num_superclasses, drop_rate=drop_rate,
        drop_path_rate=drop_path_rate, **kwargs)


def dhvt_tiny_cifar_patch2(num_classes=100, num_superclasses=0, drop_rate=0.,
                           drop_path_rate=0., **kwargs):
    return VisionTransformer(
        img_size=32, patch_size=2, embed_dim=192, depth=12, num_heads=4,
        mlp_ratio=4, apply_dhvt=True, num_classes=num_classes,
        num_superclasses=num_superclasses, drop_rate=drop_rate,
        drop_path_rate=drop_path_rate, **kwargs)


def dhvt_small_cifar_patch2(num_classes=100, num_superclasses=0, drop_rate=0.,
                            drop_path_rate=0., **kwargs):
    return VisionTransformer(
        img_size=32, patch_size=2, embed_dim=384, depth=12, num_heads=8,
        mlp_ratio=4, apply_dhvt=True, num_classes=num_classes,
        num_superclasses=num_superclasses, drop_rate=drop_rate,
        drop_path_rate=drop_path_rate, **kwargs)


def dhvt_tiny_imagenet_patch16(num_classes=1000, num_superclasses=0, drop_rate=0.,
                               drop_path_rate=0., **kwargs):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3,
        mlp_ratio=4, apply_dhvt=True, num_classes=num_classes,
        num_superclasses=num_superclasses, drop_rate=drop_rate,
        drop_path_rate=drop_path_rate, **kwargs)


def dhvt_small_imagenet_patch16(num_classes=1000, num_superclasses=0, drop_rate=0.,
                                drop_path_rate=0., **kwargs):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6,
        mlp_ratio=4, apply_dhvt=True, num_classes=num_classes,
        num_superclasses=num_superclasses, drop_rate=drop_rate,
        drop_path_rate=drop_path_rate, **kwargs)
