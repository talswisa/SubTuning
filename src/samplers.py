from torch.utils.data.sampler import Sampler
import numpy as np


class ActiveLearningSampler(Sampler):

    def __init__(self, indices) -> None:
        self.indices = indices
        np.random.shuffle(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
