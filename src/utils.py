from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer


def filter_config(config: DictConfig) -> dict:

    def is_special_key(key: str) -> bool:
        return key[0] == "_" and key[-1] == "_"

    filt = {
        k: v
        for k, v in config.items()
        if (not OmegaConf.is_interpolation(config, k)) and (not is_special_key(k)) and v is not None
    }

    return filt


def log_hyperparams(config: DictConfig, trainer: Trainer) -> None:
    hparams = {}
    # choose which parts of hydra config will be saved to loggers as hyperparameters
    for key in ["trainer", "model", "datamodule"]:
        hparams[key] = filter_config(config[key])

    # add paramters in conf/config.yaml
    for k, v in config.items():
        if not isinstance(v, DictConfig):
            hparams[k] = v

    trainer.logger.log_hyperparams(hparams)
