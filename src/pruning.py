from torch import nn
import torch
import wandb

try:
    import sys
    sys.path.append('/home/jovyan/finetune/Torch-Pruning')  # https://github.com/VainF/Torch-Pruning
    import torch_pruning as tp
except:
    print("can't import torch pruning, please clone branch v.0.2.8 from https://github.com/VainF/Torch-Pruning, and add it to your system path.")


def prune_resnet(model, amount_to_prune=0.5, pruner_type='local', pruning_norm=1, *args, **kwargs):
    """
    Prunes layer4 of ResNet-50
    """
    print_model_to_file(model, 'orig_model.txt')
    wandb.save('orig_model.txt')
    # the pruning library does not work well with the head so we are temporary removing in, will create a new head after the pruning
    model.head = nn.Identity()
    layer = getattr(model.backbone, "layer4")
    prune_model(layer, ch_sparsity=amount_to_prune, pruner_type=pruner_type, pruning_norm=pruning_norm)

    # create a new head with the correct input size
    last_layer = getattr(model.backbone, "layer4")[-1]
    output_size = last_layer.conv3.out_channels
    model.representation_size = output_size
    model.head = model.get_mlp()

    print_model_to_file(model, 'pruned_model.txt')
    wandb.save('pruned_model.txt')
    return model


def prune_model(layer, ch_sparsity=0.5, pruning_norm=1, pruner_type='local'):
    orig_params = sum(p.numel() for p in layer.parameters())
    n_blocks = len(layer)
    resolution = 14 * (2**(n_blocks - 1))
    in_channels = 1024
    example_inputs = torch.randn(1, in_channels, resolution, resolution)

    print("defining pruner")
    if pruner_type == 'local':
        pruner = tp.pruner.LocalMagnitudePruner(
            layer,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(p=pruning_norm),
            total_steps=1,  # number of iterations
            ch_sparsity=ch_sparsity,  # channel sparsity
            ignored_layers=[],  # ignored_layers will not be pruned
        )
    elif pruner_type == 'global':
        pruner = tp.pruner.GlobalMagnitudePruner(
            layer,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(p=pruning_norm),
            total_steps=1,  # number of iterations
            max_ch_sparsity=0.9,
            ch_sparsity=ch_sparsity,  # channel sparsity
            ignored_layers=[],  # ignored_layers will not be pruned
            )
    else:
        raise ("only local and global pruner_type are supported")
    pruner.step()
    print(f'new layer params: {sum(p.numel() for p in layer.parameters()):,d} old layer params: {orig_params:,d}')


def print_model_to_file(model, file_path):
    original_stdout = sys.stdout
    with open(file_path, 'w') as sys.stdout:
        print(model)
    sys.stdout = original_stdout
    print(f"saved {file_path}")