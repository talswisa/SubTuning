from torchvision import models
import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


def transform_conv(conv, is_first=False):
    if is_first:
        new_conv = nn.Conv2d(conv.in_channels,
                             2 * conv.out_channels,
                             conv.kernel_size,
                             conv.stride,
                             conv.padding,
                             bias=conv.bias is not None)
        new_conv.weight.data[:conv.weight.data.shape[0]] = conv.weight.data
        new_conv.weight.data[conv.weight.data.shape[0]:] = conv.weight.data
    else:
        new_conv = nn.Conv2d(2 * conv.in_channels,
                             2 * conv.out_channels,
                             conv.kernel_size,
                             conv.stride,
                             conv.padding,
                             bias=conv.bias is not None,
                             groups=2)
        new_conv.weight.data[:conv.weight.data.shape[0]] = conv.weight.data
        new_conv.weight.data[conv.weight.data.shape[0]:] = conv.weight.data
    return new_conv


def transform_bn(bn):
    new_bn = nn.BatchNorm2d(2 * bn.num_features)
    new_bn.weight.data[:bn.weight.data.shape[0]] = bn.weight.data
    new_bn.weight.data[bn.weight.data.shape[0]:] = bn.weight.data
    new_bn.bias.data[:bn.bias.data.shape[0]] = bn.bias.data
    new_bn.bias.data[bn.bias.data.shape[0]:] = bn.bias.data
    new_bn.running_mean.data[:bn.running_mean.data.shape[0]] = bn.running_mean.data
    new_bn.running_mean.data[bn.running_mean.data.shape[0]:] = bn.running_mean.data
    new_bn.running_var.data[:bn.running_var.data.shape[0]] = bn.running_var.data
    new_bn.running_var.data[bn.running_var.data.shape[0]:] = bn.running_var.data
    return new_bn


def transform_linear(linear):
    new_linear = nn.Linear(2 * linear.in_features,
                           linear.out_features,
                           bias=linear.bias is not None)
    new_linear.weight.data[:, :linear.weight.data.shape[1]] = linear.weight.data / 2
    new_linear.weight.data[:, linear.weight.data.shape[1]:] = linear.weight.data / 2
    if linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear


def transform_layer(layer, is_first_block=False):
    new_layer = deepcopy(layer)
    for i, (name, module) in enumerate(layer.named_children()):
        is_first = i == 0 and is_first_block
        if isinstance(module, nn.Conv2d):
            new_layer._modules[name] = transform_conv(module, is_first=is_first)
        elif isinstance(module, nn.BatchNorm2d):
            new_layer._modules[name] = transform_bn(module)
        elif hasattr(module, '_modules'):
            new_layer._modules[name] = transform_layer(module,
                                                       is_first_block=is_first_block and
                                                       (name in ['0', 'downsample']))
    return new_layer


class Concatify(nn.Module):
    """
        This PyTorch module doubles the number of channels in a tensor by concatenating the input
        with itself along the channel dimension.
        Forwarding the resulting tensor through a convolution with two groups is equivalnet to running
        it through two parallel layers - the original frozen layer and a new finetuned layer.
    """

    def forward(self, x):
        return torch.cat([x, x], dim=1)


class Batchify(nn.Module):
    """
        This PyTorch module halves the number of channels in a tensor and doubles the batch size by
        reshaping the input. One half represents the output of the frozen layer, and the other half
        represents the output of the new layer.
        The output of the frozen layer is concatenated with the output of the new layer, and the resulting
        tensor is passed to the following frozen layers.
    """

    def forward(self, x):
        return x.reshape(x.shape[0] * 2, x.shape[1] // 2, x.shape[2], x.shape[3])


class BatchInferenceResNet(nn.Module):

    def __init__(self, backbone, layer_to_finetune1, layer_to_finetune2=None):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

        if layer_to_finetune2 is not None and layer_to_finetune1.split(
                '_')[0] == layer_to_finetune2.split('_')[0]:
            self.handle_2_blocks_from_same_layer(layer_to_finetune1, layer_to_finetune2)
        else:
            self.handle_layers_to_finetune(layer_to_finetune1)
            if layer_to_finetune2:
                self.handle_layers_to_finetune(layer_to_finetune2)

    def handle_layers_to_finetune(self, layer_to_finetune):
        layer = layer_to_finetune.split('_')[0]
        block = int(layer_to_finetune.split('_')[1])

        layer_obj = getattr(self, layer)
        layers = list(layer_obj.children())  # blocks
        layers[block] = transform_layer(layers[block], is_first_block=False)
        layers.insert(block, Concatify())
        layers.insert(block + 2, Batchify())  # +2 because of concat
        setattr(self, layer, nn.Sequential(*layers))

    def handle_2_blocks_from_same_layer(self, layer_to_finetune1, layer_to_finetune2):
        layer1 = layer_to_finetune1.split('_')[0]

        block1 = int(layer_to_finetune1.split('_')[1])
        block2 = int(layer_to_finetune2.split('_')[1])

        layer_obj1 = getattr(self, layer1)

        layers = list(layer_obj1.children())
        layers[block1] = transform_layer(layers[block1], is_first_block=False)
        layers[block2] = transform_layer(layers[block2], is_first_block=False)
        layers.insert(block1, Concatify())
        layers.insert(block2 + 2, Batchify())  # +2 because of concat todo
        setattr(self, layer1, nn.Sequential(*layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x




def measure_inf_time(net, bs=1):
    net.cuda()
    net.eval()
    dummy_input = torch.randn(bs, 3, 224, 224, dtype=torch.float).cuda()
    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    torch.cuda.empty_cache()
    # MEASURE PERFORMANCE
    with torch.no_grad():
        #GPU-WARM-UP
        for _ in range(repetitions):
            _ = net(dummy_input)
        for rep in range(repetitions):
            starter.record()
            _ = net(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    return timings


if __name__ == "__main__":
    backbone = models.resnet50(num_classes=1000).eval()
    params = []
    layers = [
        'layer1_0', 'layer1_1', 'layer1_2', 'layer2_0', 'layer2_1', 'layer2_2', 'layer2_3',
        'layer3_0', 'layer3_1', 'layer3_2', 'layer3_3', 'layer3_4', 'layer3_5', 'layer4_0',
        'layer4_1', 'layer4_2'
    ]
    # one block:
    bbs = [backbone] + [BatchInferenceResNet(backbone, layer_to_finetune1=f'{l}') for l in layers]

    inf_times_means = []
    inf_times_stds = []

    for bb in bbs:
        bb = torch.jit.script(bb)
        curr_times = []
        for i in range(10):
            inf_time = measure_inf_time(bb, bs=1)
            curr_times.append(np.median(inf_time))

        inf_times_means.append(curr_times)
        print(np.median(curr_times))

    inf_times_means = np.array(inf_times_means).squeeze()
    print(f"one block inf times means: {inf_times_means}")
    # two blocks:
    layers = list(zip(layers[:-1], layers[1:]))

    bbs = [backbone] + [
        BatchInferenceResNet(backbone, layer_to_finetune1=l1, layer_to_finetune2=l2)
        for l1, l2 in layers
    ]

    inf_times_means = []
    inf_times_stds = []

    for bb in bbs:
        bb = torch.jit.script(bb)
        curr_times = []
        for i in range(10):
            inf_time = measure_inf_time(bb, bs=1)
            curr_times.append(np.median(inf_time))

        inf_times_means.append(curr_times)
        print(np.median(curr_times))

    inf_times_means = np.array(inf_times_means).squeeze()
    print(f"two blocks inf times means: {inf_times_means}")