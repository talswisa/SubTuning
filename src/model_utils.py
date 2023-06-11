import torch
import torch.nn as nn
from copy import deepcopy


def get_mlp(input_size, output_size, width, depth):
    layers = []
    in_width = input_size
    for _ in range(depth - 1):
        layers.append(nn.Linear(in_width, width, bias=False))
        layers.append(nn.ReLU(inplace=True))
        in_width = width
    layers.append(nn.Linear(in_width, output_size, bias=True))
    return nn.Sequential(*layers)


class SelectiveFinetune(nn.Module):

    def __init__(self, backbone, head_params, layers_to_finetune):
        super().__init__()
        self.backbone = backbone
        self.head_params = head_params
        self.layers_to_finetune = layers_to_finetune
        self.representation_size = self.get_representation_size()
        self.disable_original_head()
        self.head = self.get_mlp()
        self.toggle_grad()

    def get_mlp(self):
        mlp_in_channels = self.representation_size
        depth = self.head_params.get('depth', 1)
        width = self.head_params.get('width', mlp_in_channels)
        num_classes = self.head_params['num_classes']

        mlp = get_mlp(mlp_in_channels, num_classes, width, depth)
        return mlp

    def forward(self, x):
        is_training = self.training
        self.backbone.eval()
        if is_training:
            for layer in self.get_unfrozen_layers():
                layer.train()

        x = self.backbone(x)

        x = self.head(x)
        return x

    def toggle_grad(self):
        raise NotImplementedError

    def get_unfrozen_layers(self):
        raise NotImplementedError

    def get_representation_size(self):
        raise NotImplementedError

    def disable_original_head(self):
        raise NotImplementedError


class SelectiveFinetuneResNet(SelectiveFinetune):

    def __init__(self, backbone, head_params, layers_to_finetune=None):
        super().__init__(backbone, head_params, layers_to_finetune)

    def get_representation_size(self):
        return self.backbone.fc.in_features

    def disable_original_head(self):
        self.backbone.fc = nn.Identity()

    def toggle_grad(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for layer in self.get_unfrozen_layers():
            for param in layer.parameters():
                param.requires_grad = True

    def get_unfrozen_layers(self):
        for layer_name in self.layers_to_finetune:
            assert layer_name.split("_")[0] in [
                'conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4'
            ], f"layer_name={layer_name} is not valid"
            if "_" in layer_name:
                layer_name, block = layer_name.split("_")
                block = int(block)
                layer = getattr(self.backbone, layer_name)
                assert block in range(len(layer)), f"block={block} is not valid"
                layer = layer[block]
            else:
                layer = getattr(self.backbone, layer_name)
            yield layer


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LoraAttention(nn.Module):

    def __init__(self, attention_block, rank=2):
        super().__init__()
        self.orig_attention = attention_block
        self.dim = attention_block.qkv.in_features
        qkv_has_bias = attention_block.qkv.bias is not None
        proj_has_bias = attention_block.proj.bias is not None
        self.rank = rank
        # initialize As with gaussian and Bs with zeros
        self.delta_kqv_A = nn.Linear(self.dim, self.rank * 3, bias=qkv_has_bias)

        self.delta_kqv_B = nn.Linear(self.rank * 3, self.dim * 3,
                                     bias=qkv_has_bias).apply(lambda m: nn.init.zeros_(m.weight))

        self.delta_wo_A = nn.Linear(self.dim, self.rank, bias=proj_has_bias)

        self.delta_wo_B = nn.Linear(self.rank, self.dim, bias=proj_has_bias)
        nn.init.zeros_(self.delta_wo_B.weight)
        if proj_has_bias:
            nn.init.zeros_(self.delta_wo_B.bias)

    def toggle_grad(self):
        for param in self.orig_attention.parameters():
            param.requires_grad = False

        for param in self.delta_kqv_A.parameters():
            param.requires_grad = True

        for param in self.delta_kqv_B.parameters():
            param.requires_grad = True

        for param in self.delta_wo_A.parameters():
            param.requires_grad = True

        for param in self.delta_wo_B.parameters():
            param.requires_grad = True

    def forward(self, x):
        B, N, C = x.shape
        orig_qkv = self.orig_attention.qkv(x).reshape(B, N, 3, self.orig_attention.num_heads,
                                                      C // self.orig_attention.num_heads).permute(
                                                          2, 0, 3, 1, 4)
        q, k, v = orig_qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        delta_kqv_A = self.delta_kqv_A(x)
        delta_kqv = self.delta_kqv_B(delta_kqv_A)
        delta_kqv = delta_kqv.reshape(B, N, 3, self.orig_attention.num_heads,
                                      C // self.orig_attention.num_heads).permute(2, 0, 3, 1, 4)
        delta_q, delta_k, delta_v = delta_kqv.unbind(0)
        # devide delta by rank:
        delta_q, delta_k, delta_v = delta_q * 768 / self.rank, delta_k * 768 / self.rank, delta_v * 768 / self.rank
        q, k, v = q + delta_q, k + delta_k, v + delta_v

        attn = (q @ k.transpose(-2, -1)) * self.orig_attention.scale
        attn = attn.softmax(dim=-1)
        attn = self.orig_attention.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # Update Wo with low-rank matrix
        delta_wo_A = self.delta_wo_A(x)
        delta_wo = self.delta_wo_B(delta_wo_A)
        # devide delta by rank:
        delta_wo = delta_wo * 768 / self.rank
        x = self.orig_attention.proj(x) + delta_wo

        x = self.orig_attention.proj_drop(x)
        return x


class SelectiveFinetuneViT(SelectiveFinetune):

    def __init__(self, backbone, head_params, layers_to_finetune, layers_ranks=None):
        print(f"layers_to_finetune={layers_to_finetune}")
        print(f"layers_ranks={layers_ranks}")
        self.layers_ranks = layers_ranks
        super().__init__(backbone, head_params, layers_to_finetune)

    def get_representation_size(self):
        return self.backbone.head.in_features

    def get_unfrozen_layers(self):
        for block_index in self.layers_to_finetune:
            yield self.backbone.blocks[block_index]

    def replace_attentions(self):
        random_input = torch.randn(1, 3, 224, 224)
        output_before_change = self.backbone(random_input)
        if self.layers_ranks > 0:
            for block_idx in self.layers_to_finetune:
                block = self.backbone.blocks[block_idx]
                print(
                    f"replacing attention in block {block_idx} with lora block with rank {self.layers_ranks}"
                )
                block.attn = LoraAttention(block.attn, self.layers_ranks)
        output_after_change = self.backbone(random_input)
        assert torch.allclose(output_before_change, output_after_change)

    def toggle_grad(self):
        self.replace_attentions()
        for param in self.backbone.parameters():
            param.requires_grad = False

        for block in self.get_unfrozen_layers():
            block.requires_grad = True
            for param in block.parameters():
                param.requires_grad = True

            if isinstance(block.attn, LoraAttention):
                block.attn.toggle_grad()

        self.head.requires_grad = True

    def disable_original_head(self):
        self.backbone.head = nn.Identity()
