# Less is More: Selective Layer Finetuning with SubTuning

[[Paper]](https://arxiv.org/abs/2302.06354)

SubTuning is a parameter-efficient method for fine-tuning pretrained neural networks. It selectively trains specific layers while keeping others at their pre-trained state, optimizing performance based on the task and data distribution. Compared to conventional fine-tuning, SubTuning excels in scenarios with scarce or corrupted data and often matches the performance of traditional fine-tuning with abundant data. It has been successfully applied across various tasks, architectures, and pre-training methods.
## Approach
![SubTuning](SubTuning.png)

## Usage
To run experiments, follow these steps:

1. Clone the repository.
2. Build and run the Docker container by running the following commands:
```
cd docker
docker build -t subtuningimage .
docker run --rm -it subtuningimage
```
3. insert your WandB entity, API key, and project name in the wandb.text file located in the project directory.

4.  run experiments using the provided bash scripts. The hyperparameters in the bash scripts can be modified. For instance, by changing the value of _model.layers_to_finetune_, you can choose the layers to fine-tune.
For example, to create a finetune profile for ResNet50, run the following command:
```
bash scripts/greedy_subtuning_vit_cifar100.sh
```

To create the VTAB-1k data, run the `get_pytorch_versions_of_vtab.py` script.


If you have any issues or questions, please create a git issue.

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2302.06354,
  doi = {10.48550/ARXIV.2302.06354},
  url = {https://arxiv.org/abs/2302.06354},
  author = {Kaplun, Gal and Gurevich, Andrey and Swisa, Tal and David, Mazor and Shalev-Shwartz, Shai and Malach, Eran},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {SubTuning: Efficient Finetuning for Multi-Task Learning},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```