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

    def __init__(self,
                 backbone,
                 head_params,
                 layers_to_finetune=None,
                 use_freeze_bb=False,
                 reinit_layers=False):
        super().__init__()
        self.backbone = backbone
        self.frozen_backbone = None
        self.head_params = head_params
        self.layers_to_finetune = layers_to_finetune
        self.use_frozen_bb = use_freeze_bb
        self.reinit_layers = reinit_layers
        self.original_representation_size = self.get_representation_size()
        if self.use_frozen_bb:
            self.representation_size = self.original_representation_size * 2
        else:
            self.representation_size = self.original_representation_size

        assert self.layers_to_finetune is not None, "Must specify layers to finetune"
        self.handle_frozen_backbone()
        self.disable_original_head()
        self.head = self.get_mlp()

    def get_mlp(self):
        mlp_in_channels = self.representation_size
        depth = self.head_params.get('depth', 1)
        width = self.head_params.get('width', mlp_in_channels)
        num_classes = self.head_params['num_classes']

        mlp = get_mlp(mlp_in_channels, num_classes, width, depth)
        return mlp

    @staticmethod
    def reinit_layer(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    def freeze_backbone_and_reinit(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for layer in self.get_unfrozen_layers():
            for param in layer.parameters():
                if self.reinit_layers:
                    layer.apply(self.reinit_layer)
                param.requires_grad = True

        if self.frozen_backbone is not None:
            for param in self.frozen_backbone.parameters():
                param.requires_grad = False

    def duplicate_backbone(self):
        self.frozen_backbone = deepcopy(self.backbone)

    def forward(self, x):
        is_training = self.training
        self.backbone.eval()
        if is_training:
            for layer in self.get_unfrozen_layers():
                layer.train()

        if self.use_frozen_bb:
            x1 = self.backbone(x)
            x2 = self.frozen_backbone(x)
            x = torch.cat((x1, x2), dim=1)
        else:
            x = self.backbone(x)

        x = self.head(x)
        return x

    def handle_frozen_backbone(self):
        raise NotImplementedError

    def get_unfrozen_layers(self):
        raise NotImplementedError

    def get_representation_size(self):
        raise NotImplementedError

    def disable_original_head(self):
        raise NotImplementedError


class SelectiveFinetuneResNet(SelectiveFinetune):

    def __init__(self,
                 backbone,
                 head_params,
                 layers_to_finetune=None,
                 use_freeze_bb=False,
                 reinit_layers=False):
        super().__init__(backbone, head_params, layers_to_finetune, use_freeze_bb, reinit_layers)

    def get_representation_size(self):
        return self.backbone.fc.in_features

    def disable_original_head(self):
        self.backbone.fc = nn.Identity()
        if self.use_frozen_bb:
            self.frozen_backbone.fc = nn.Identity()

    def handle_frozen_backbone(self):
        if self.freeze_backbone_and_reinit:
            self.duplicate_backbone()
        self.freeze_backbone_and_reinit()
        ## Save weights for comparison later
        self.saved_first_conv_weights = self.backbone.conv1.weight.clone().detach().cpu()
        self.saved_first_bn_weights = self.backbone.bn1.weight.clone().detach().cpu()
        self.saved_first_bn_bias = self.backbone.bn1.bias.clone().detach().cpu()
        self.saved_first_bn_running_mean = self.backbone.bn1.running_mean.clone().detach().cpu()
        self.saved_first_bn_running_var = self.backbone.bn1.running_var.clone().detach().cpu()

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


class SelectiveFinetuneViT(SelectiveFinetune):

    def __init__(self,
                 backbone,
                 head_params,
                 layers_to_finetune=None,
                 use_freeze_bb=False,
                 reinit_layers=True):
        super().__init__(backbone, head_params, layers_to_finetune, use_freeze_bb, reinit_layers)

    def get_representation_size(self):
        return self.backbone.heads.head.in_features

    def disable_original_head(self):
        self.backbone.heads.head = nn.Identity()
        if self.use_frozen_bb:
            self.frozen_backbone.heads.head = nn.Identity()

    def handle_frozen_backbone(self):
        if self.use_frozen_bb:
            self.duplicate_backbone()
        self.freeze_backbone_and_reinit()

        self.saved_first_attention_weights = (self.backbone.encoder.layers.encoder_layer_0.
                                              self_attention.out_proj.weight.clone().detach().cpu())
        self.saved_first_mlp_weights = (
            self.backbone.encoder.layers.encoder_layer_0.mlp[0].weight.clone().detach().cpu())
        self.saved_first_bn_weights = (
            self.backbone.encoder.layers.encoder_layer_0.ln_1.weight.clone().detach().cpu())
        self.saved_first_bn_bias = (
            self.backbone.encoder.layers.encoder_layer_0.ln_1.bias.clone().detach().cpu())

    def get_unfrozen_layers(self):
        for layer_name in self.layers_to_finetune:
            vit_layers = [f'encoder_layer_{i}' for i in range(12)]
            assert layer_name in vit_layers, f"layer_name={layer_name} is not valid"
            layer = getattr(self.backbone.encoder.layers, layer_name)
            yield layer
