import hydra
from omegaconf import DictConfig
from src.train import train
from src.greedy_subset_selection import find_best_layers
from src.lora_search import find_best_lora


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    if config.greedy_subset_selection:

        best_layers = find_best_layers(config)
        config.model.layers_to_finetune = best_layers

        if config.datamodule.vtab:
            assert not config.datamodule.vtab_complete, "searched for best layers with vtab_complete=True"
            config.datamodule.vtab_complete = True  # train on 1k, test on all test
        config.datamodule.cross_val_index = None
        config.to_report = True
        # running 5 times to get mean acc with the best layers
        for _ in range(5):
            train(config)
    elif config.find_best_lora:
        best_rank = find_best_lora(config)
        config.model.layers_ranks = best_rank
        if config.datamodule.vtab:
            assert not config.datamodule.vtab_complete, "searched for best lora with vtab_complete=True"
            config.datamodule.vtab_complete = True  # train on 1k, test on all test
        config.datamodule.cross_val_index = None
        config.to_report = True
        # running 5 times to get mean acc with the best layers
        for _ in range(5):
            train(config)
    else:
        train(config)


if __name__ == "__main__":
    main()
