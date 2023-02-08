import hydra
from omegaconf import DictConfig
from src.train import train


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    train(config)


if __name__ == "__main__":
    main()
