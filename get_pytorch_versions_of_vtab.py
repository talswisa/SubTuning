import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_datasets as tfds
from torchvision.transforms import transforms
from collections import Counter


def create_vtab_version(dataset_name, version):
    # Load the dataset

    dataset_builder = tfds.builder(f"{dataset_name}:{version}.*.*")
    dataset_builder.download_and_prepare()
    # Calculate the number of training examples to use for train800 and val200 splits
    train800_examples = 800
    val200_examples = 200

    # Calculate the start index for the val200 split based on the 90% split
    total_train_examples = dataset_builder.info.splits["train"].num_examples
    val200_start_index = int(total_train_examples * 0.8)

    # Define the dataset splits
    splits = {
        "train800": f"train[:{train800_examples}]",
        "val200": f"train[{val200_start_index}:{val200_start_index + val200_examples}]",
    }

    # Load the train800 and val200 splits
    train800_set = tfds.load(f"{dataset_name}:{version}.*.*",
                             split=splits["train800"],
                             as_supervised=True)

    val200_set = tfds.load(f"{dataset_name}:{version}.*.*",
                           split=splits["val200"],
                           as_supervised=True)
    test_set = tfds.load(f"{dataset_name}:{version}.*.*", split="test", as_supervised=True)

    # Step 1: Convert the TensorFlow datasets to PyTorch tensors
    def tfds_to_pytorch_tensors(tfds_set):
        images, labels = [], []
        resize = transforms.Resize((224, 224), antialias=True)
        label_counter = Counter()
        for i, (image, label) in enumerate(tfds_set):
            # permute from (height, width, channels) to (channels, height, width)
            image = torch.tensor(image.numpy())
            image = image.permute(2, 0, 1)
            image = resize(image)
            assert image.shape == (3, 224, 224)
            images.append(image)
            label = torch.tensor(label.numpy())
            labels.append(label)
            label_counter[label.item()] += 1

        print(f"labels: {label_counter}")
        print(f"n_labels: {len(label_counter)}")

        images = torch.stack(images)
        labels = torch.stack(labels)

        # normalize to [0, 1]
        images = images.float() / 255.0

        return images, labels

    train800_images, train800_labels = tfds_to_pytorch_tensors(train800_set)
    val200_images, val200_labels = tfds_to_pytorch_tensors(val200_set)
    test_images, test_labels = tfds_to_pytorch_tensors(test_set)

    # compute the mean and std of the train800 set for transforms.Normalize
    mean = train800_images.float().mean(dim=(0, 2, 3))
    std = train800_images.float().std(dim=(0, 2, 3))

    print(f"mean: {mean}")
    print(f"std: {std}")

    # save tensors to disk
    # create folder if it doesn't exist
    os.makedirs(f"data/vtab/{dataset_name}", exist_ok=True)
    torch.save(train800_images, f"data/vtab/{dataset_name}/train800_images.pt")
    torch.save(train800_labels, f"data/vtab/{dataset_name}/train800_labels.pt")
    torch.save(val200_images, f"data/vtab/{dataset_name}/val200_images.pt")
    torch.save(val200_labels, f"data/vtab/{dataset_name}/val200_labels.pt")
    torch.save(test_images, f"data/vtab/{dataset_name}/test_images.pt")
    torch.save(test_labels, f"data/vtab/{dataset_name}/test_labels.pt")


create_vtab_version("cifar100", 3)
create_vtab_version("caltech101", 3)
create_vtab_version("oxford_flowers102", 2)
create_vtab_version("dmlab", 2)
