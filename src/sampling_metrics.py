import torch
import numpy as np


def get_max_probs(probs, label_mask=None, return_indices=False):
    # returns the maximum class probability for each example, excluding the classes that are masked for each example
    if label_mask is not None:
        probs = probs - torch.eye(probs.shape[1])[label_mask]

    max_probs, max_indices = torch.max(probs, dim=1)
    if return_indices:
        return max_probs, max_indices
    else:
        return max_probs


def unsupervised_margin(logits: torch.Tensor):
    # maximum prob - second maximum prob
    probs = torch.softmax(logits, dim=1)
    maximum_probs, maximum_indices = get_max_probs(probs, return_indices=True)
    second_maximum_probs = get_max_probs(probs, maximum_indices)
    diff = maximum_probs - second_maximum_probs
    margin = torch.abs(diff)
    return margin
