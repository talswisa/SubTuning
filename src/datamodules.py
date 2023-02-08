import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, INaturalist, StanfordCars
from torchvision.transforms import transforms
from src.samplers import ActiveLearningSampler
import numpy as np


SUPPORTED_DATASETS_CLASSES = {'cifar10': 10, 'cifar100': 100, 'flowers102': 102, 'inaturalist': 10000}


def get_dataset(data_path, train, dataset, input_size=224):
    if dataset.lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        return CIFAR10(data_path, train=train, transform=get_transforms(mean, std, train, input_size), download=True)
    elif dataset.lower() == "cifar100":
        mean, std = (0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)
        return CIFAR100(data_path, train=train, transform=get_transforms(mean, std, train, input_size), download=True)
        # mean, std = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    elif "flowers" in dataset.lower():
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        # Flowers has different API than CIFAR (CIFAR uses "train: bool" but flowers users "split: str")
        return Flowers102(data_path, split="train" if train else "val", transform=get_transforms(mean, std, train, input_size), download=True)
    elif 'inaturalist' in dataset.lower():
        # INaturalist has different API than CIFAR/Flowers (CIFAR uses "train: bool" and flowers users "split: str", but INaturalist is special and uses "version: str")
        # the large INaturalist dataset contains 2.7M examples, the train_mini dataset contains 500k images with 50 examples per class.
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        version = "2021_" + ("train_mini" if train else "valid")
        try:
            ds = INaturalist(data_path, version=version, transform=get_transforms(mean, std, train, input_size), download=True)
        except:
            ds = INaturalist(data_path, version=version, transform=get_transforms(mean, std, train, input_size), download=False)
        return ds
    elif 'cars' in dataset.lower():
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        return StanfordCars(data_path, split="train" if train else "test", transform=get_transforms(mean, std, train, input_size), download=True)

def get_transforms(mean, std, train, input_size=224):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform

def get_flowers_transforms(mean, std, train, input_size=224):
    if train:
        transform = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform

def get_initial_indices(dataset, query_budget):
    """
    Returns a list of indices of the initial training set for active learning.
    """
    try:
        query_budget = query_budget[0]
    except TypeError:
        pass  # query_budget is already an int

    indices = np.random.choice(len(dataset), query_budget, replace=False)

    return indices


class DataModule(pl.LightningDataModule):

    def __init__(
            self,
            data_dir: str = None,
            batch_size: int = 256,
            num_workers: int = 4,
            prefetch_factor: int = 2,
            dataset: str = "cifar10",
            do_active_learning: bool = False,
            query_budget = None,
            input_size: int = 224,
            n_examples_in_subset=None,  # chec
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.prefetch_factor = prefetch_factor
        self.input_size = input_size
        self.do_active_learning = do_active_learning
        self.query_budget = query_budget
        self.sampler = None
        self.n_examples_in_subset = n_examples_in_subset

    def setup(self, stage=None):
        assert self.dataset.lower() in SUPPORTED_DATASETS_CLASSES, f"Only {SUPPORTED_DATASETS_CLASSES.keys()} are supported {self.dataset} were provided"
        self.train_set = get_dataset(data_path=self.data_dir, train=True, dataset=self.dataset, input_size=self.input_size)
        self.val_set = get_dataset(data_path=self.data_dir, train=False, dataset=self.dataset, input_size=self.input_size)

        if self.n_examples_in_subset is not None:
            print(f"Training with a random subset of {self.n_examples_in_subset} examples")
            indices = np.random.choice(len(self.train_set), self.n_examples_in_subset, replace=False)
            self.train_set = Subset(self.train_set, indices)

        if self.do_active_learning:
            initial_indices = get_initial_indices(self.train_set, self.query_budget)
            self.sampler = ActiveLearningSampler(initial_indices)

    def train_dataloader(self):
        print("loading train dataloader")

        if self.sampler is not None:
            shuffle = False
        else:
            shuffle = True

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            sampler=self.sampler,
            shuffle=shuffle)

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
        )
