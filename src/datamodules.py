import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import CIFAR10, CIFAR100, Flowers102, INaturalist, StanfordCars
from torchvision.transforms import transforms
from collections import Counter
from src.samplers import ActiveLearningSampler
import numpy as np
import torch
import os

SUPPORTED_DATASETS_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'flowers102': 102,
    'inaturalist': 10000,
    'cars': 196,
    'caltech101': 102,  # 101 + background class
    'dmlab': 6,
}


def get_mean_std(dataset, vtab):
    if 'cifar10' in dataset.lower():
        if vtab:
            return (0.4906, 0.4808, 0.4492), (0.2489, 0.2455, 0.2622)
        else:
            return (0.5053, 0.4862, 0.4430), (0.2673, 0.2550, 0.2776)
    elif 'cifar100' in dataset.lower():
        if vtab:
            return (0.5053, 0.4862, 0.4430), (0.2673, 0.2550, 0.2776)
        else:
            return (0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)
    elif 'flowers' in dataset.lower():
        if vtab:
            return (0.4287, 0.3838, 0.2964), (0.2922, 0.2437, 0.2729)
        else:
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif 'caltech101' in dataset.lower():
        assert vtab
        return (0.5326, 0.5132, 0.4730), (0.3208, 0.3166, 0.3269)
    elif 'dmlab' in dataset.lower():
        assert vtab
        return (0.4951, 0.6012, 0.6065), (0.2211, 0.1998, 0.3259)


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


def get_cross_val_dataset(dataset: torch.utils.data.Dataset, cross_val_index):
    """
    Splits the dataset into 5 folds and returns all but the i'th fold for training
    or the i'th fold for validation.
    """
    indices = np.arange(len(dataset))
    shuffled_indices = shuffle_arrays(indices)[0]

    fold_size = len(dataset) // 5

    train_indices = np.concatenate([
        shuffled_indices[:cross_val_index * fold_size],
        shuffled_indices[(cross_val_index + 1) * fold_size:]
    ])

    val_indices = shuffled_indices[cross_val_index * fold_size:(cross_val_index + 1) * fold_size]
    return torch.utils.data.Subset(dataset,
                                   train_indices), torch.utils.data.Subset(dataset, val_indices)


def shuffle_arrays(*arrays):
    # Save the current randomness state
    numpy_state = np.random.get_state()

    # Set a seed
    np.random.seed(42)

    length = len(arrays[0])
    for array in arrays:
        assert len(array) == length, "All arrays must have the same length"

    indices = np.arange(length)
    np.random.shuffle(indices)

    shuffled_arrays = []
    for array in arrays:
        if isinstance(array, np.ndarray):
            shuffled_arrays.append(array[indices])
        elif isinstance(array, torch.Tensor):
            shuffled_arrays.append(array[torch.from_numpy(indices)])

    # Restore the previous randomness state
    np.random.set_state(numpy_state)

    return shuffled_arrays


def get_non_vtab_dataset(data_path, train, dataset, input_size=224):
    if dataset.lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ds = CIFAR10(data_path,
                     train=train,
                     transform=get_transforms(mean, std, train, input_size),
                     download=True)
    elif dataset.lower() == "cifar100":
        mean, std = (0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025)
        ds = CIFAR100(data_path,
                      train=train,
                      transform=get_transforms(mean, std, train, input_size),
                      download=True)
    elif "flowers" in dataset.lower():
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        # Flowers has different API than CIFAR (CIFAR uses "train: bool" but flowers users "split: str")
        ds = Flowers102(data_path,
                        split="train" if train else "val",
                        transform=get_transforms(mean, std, train, input_size),
                        download=True)
    elif 'inaturalist' in dataset.lower():
        # INaturalist has different API than CIFAR/Flowers (CIFAR uses "train: bool" and flowers users "split: str", but INaturalist is special and uses "version: str")
        # the large INaturalist dataset contains 2.7M examples, the train_mini dataset contains 500k images with 50 examples per class.
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        version = "2021_" + ("train_mini" if train else "valid")
        try:
            ds = INaturalist(data_path,
                             version=version,
                             transform=get_transforms(mean, std, train, input_size),
                             download=True)
        except:
            ds = INaturalist(data_path,
                             version=version,
                             transform=get_transforms(mean, std, train, input_size),
                             download=False)
    elif 'cars' in dataset.lower():
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        return StanfordCars(data_path,
                            split="train" if train else "test",
                            transform=get_transforms(mean, std, train, input_size),
                            download=True)
    return ds


def load_vtab_train_val(dataset_name):
    base_path = f"data/vtab/{dataset_name.lower()}/"
    train_images_path = f"{base_path}train800_images.pt"
    train_labels_path = f"{base_path}train800_labels.pt"
    val_images_path = f"{base_path}val200_images.pt"
    val_labels_path = f"{base_path}val200_labels.pt"
    train_images = torch.load(train_images_path)
    train_labels = torch.load(train_labels_path)
    val_images = torch.load(val_images_path)
    val_labels = torch.load(val_labels_path)

    return train_images, train_labels, val_images, val_labels


def load_vtab_complete_test(dataset_name):
    base_path = f"data/vtab/{dataset_name.lower()}/"
    test_images_path = f"{base_path}test_images.pt"
    test_labels_path = f"{base_path}test_labels.pt"
    test_images = torch.load(test_images_path)
    test_labels = torch.load(test_labels_path)
    return test_images, test_labels


def get_cross_val_split_from_arrays(images, labels, cross_val_index):
    # shuffle
    images, labels = shuffle_arrays(images, labels)
    # split
    images_split = torch.split(images, int(images.shape[0] / 5))
    labels_split = torch.split(labels, int(labels.shape[0] / 5))
    # get split
    train_images = torch.cat(images_split[:cross_val_index] + images_split[cross_val_index + 1:])
    train_labels = torch.cat(labels_split[:cross_val_index] + labels_split[cross_val_index + 1:])
    test_images = images_split[cross_val_index]
    test_labels = labels_split[cross_val_index]
    return train_images, train_labels, test_images, test_labels


def get_cross_val_split_from_dataset(dataset, cross_val_index):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    indices_split = np.array_split(indices, 5)
    train_indices = np.concatenate(indices_split[:cross_val_index] +
                                   indices_split[cross_val_index + 1:])
    test_indices = indices_split[cross_val_index]
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset


def get_dataset(dataset,
                input_size=224,
                vtab=False,
                vtab_complete=False,
                cross_val_index=None,
                n_examples_in_subset=None):
    if vtab:
        train_images, train_labels, test_images, test_labels = load_vtab_train_val(dataset)
        if vtab_complete:
            assert cross_val_index == None
            train_images = torch.cat((train_images, test_images))
            train_labels = torch.cat((train_labels, test_labels))
            test_images, test_labels = load_vtab_complete_test(dataset)
        else:
            if cross_val_index is not None:
                # unify
                images = torch.cat((train_images, test_images))
                labels = torch.cat((train_labels, test_labels))
                train_images, train_labels, test_images, test_labels = get_cross_val_split_from_arrays(
                    images, labels, cross_val_index)
        train_transforms = get_transforms(*get_mean_std(dataset, vtab), True, input_size)
        test_transforms = get_transforms(*get_mean_std(dataset, vtab), False, input_size)
        train_dataset = CustomDataset(train_images, train_labels, train_transforms)
        test_dataset = CustomDataset(test_images, test_labels, test_transforms)
    else:

        train_dataset = get_non_vtab_dataset(data_path="data",
                                             dataset=dataset,
                                             train=True,
                                             input_size=input_size)
        if n_examples_in_subset is not None:
            np.random.seed(42)
            print(f"Training with a random subset of {n_examples_in_subset} examples")
            rand_indices = np.random.choice(len(train_dataset), n_examples_in_subset, replace=False)
            train_dataset = Subset(train_dataset, rand_indices)
        if cross_val_index is not None:
            train_dataset, test_dataset = get_cross_val_split_from_dataset(
                train_dataset, cross_val_index)
        else:
            test_dataset = get_non_vtab_dataset(data_path="data",
                                                dataset=dataset,
                                                train=False,
                                                input_size=input_size)
    return train_dataset, test_dataset


class SafeToTensor(transforms.ToTensor):

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


def get_transforms(mean, std, train, input_size=224):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            SafeToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size), antialias=True),
            SafeToTensor(),
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
        query_budget=None,
        input_size: int = 224,
        n_examples_in_subset=None,  # chec
        vtab=False,
        vtab_complete=False,
        cross_val_index=None,
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
        self.vtab = vtab
        self.vtab_complete = vtab_complete
        self.cross_val_index = cross_val_index

    def setup(self, stage=None):
        assert self.dataset.lower(
        ) in SUPPORTED_DATASETS_CLASSES, f"Only {SUPPORTED_DATASETS_CLASSES.keys()} are supported {self.dataset} were provided"

        self.train_set, self.val_set = get_dataset(self.dataset, self.input_size, self.vtab,
                                                   self.vtab_complete, self.cross_val_index,
                                                   self.n_examples_in_subset)
        print(f"Train set size: {len(self.train_set)}")
        print(f"Val set size: {len(self.val_set)}")

        if self.do_active_learning:
            initial_indices = get_initial_indices(self.train_set, self.query_budget)
            self.sampler = ActiveLearningSampler(initial_indices)

    def train_dataloader(self):
        print("loading train dataloader")

        if self.sampler is not None:
            shuffle = False
        else:
            shuffle = True

        return DataLoader(self.train_set,
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
