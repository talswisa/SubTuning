import hydra
import wandb
import logging
import random
from pytorch_lightning.loggers import CometLogger

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import LitModel
from src.utils import log_hyperparams

log = logging.getLogger(__name__)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def get_checkpointing_callback(config):
    optional_str = ""
    # use hex to generate unique id
    random_str = hex(random.randint(0, 2**32))[2:]
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{config.model.mode}/{random_str}/",
        filename=f"{config.mode}_{optional_str}_{{epoch:02d}}",
    )
    return checkpoint_callback


def train(config: DictConfig, model=None):
    log.info(f"Instantiating logger <{config.logger._target_}>")

    current_name = f"{config.datamodule.dataset}:{config.model.arch}:{config.mode}"
    if config.mode == "finetune_layers":
        current_name += f":{config.model.layers_to_finetune}"

    logger: WandbLogger = hydra.utils.instantiate(config.logger, name=current_name)

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    if model is None:
        log.info(f"Instantiating model <{config.model._target_}>")
        model: LitModel = hydra.utils.instantiate(config.model)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpointer = get_checkpointing_callback(config)
    callbacks = [lr_monitor, checkpointer]

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=callbacks,
    )
    print(f"precision {trainer.precision}")

    log.info("Logging hyperparameters!")
    log_hyperparams(config=config, trainer=trainer)

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    wandb.finish()

    acc = trainer.callback_metrics["pred_acc"]

    return acc.item()