import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import tqdm
from src.sampling_metrics import unsupervised_margin


def compute_logits_for_dataset(dataloader, pl_module, use_precomputed_activations=False):
    all_logits = []
    is_training = pl_module.training
    pl_module.eval()
    for batch in tqdm.tqdm(dataloader):
        with torch.no_grad():
            if use_precomputed_activations:
                batch = batch.to(pl_module.device)
                logits = pl_module.model.forward_activations(batch)
            else:
                images, labels = batch
                images = images.to(pl_module.device)
                logits = pl_module.model(images)
        all_logits.append(logits)
    pl_module.train(is_training)
    all_logits = torch.cat(all_logits, dim=0)
    return all_logits.detach().cpu()


class ActiveLearning(Callback):

    def __init__(self, active_learning_params) -> None:
        super().__init__()
        self.metric = active_learning_params.metric
        self.query_budget = active_learning_params.query_budget
        self.n_epochs = active_learning_params.n_epochs
        self.n_active_learning_epoch_performed = 0

    def get_dataloader(self, datamodule, sampler=None):
        dataset = datamodule.train_set
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=datamodule.batch_size,
            shuffle=False,
            num_workers=datamodule.num_workers,
            pin_memory=True,
            sampler=sampler,
        )
        return dataloader

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.opt_max_epochs = self.n_epochs
        trainer.strategy.setup_optimizers(trainer)  # re-init optimizers with new max_epochs
        return super().on_train_start(trainer, pl_module)

    def perform_extra_epochs(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        sampler = trainer.datamodule.sampler
        print(f"starting active learning epochs with {len(sampler.indices)} examples")

        dataloader = self.get_dataloader(trainer.datamodule, sampler)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        is_training = pl_module.training
        pl_module.train()

        scheduler = pl_module.lr_schedulers()
        optimizer = trainer.optimizers[0]

        for epoch in tqdm.tqdm(range(self.n_epochs - 1)):
            for batch in dataloader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                logits = pl_module(images)
                loss = pl_module.criterion(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()

        pl_module.train(is_training)
        self.n_active_learning_epoch_performed += 1

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        preivous_indices = trainer.datamodule.sampler.indices
        try:
            curr_query_budget = self.query_budget[self.n_active_learning_epoch_performed]
        except TypeError:
            # if query_budget is not a list, it means it is a single value, so we can use it as is
            curr_query_budget = self.query_budget

        print(
            f"\nfinished active learning iteration {self.n_active_learning_epoch_performed}, \
            performing active learning example selection with query budget of size {curr_query_budget}\n")

        if self.metric == "random":
            # add self.query_budget indices that are not in the previous indices
            indices_pool = list(
                set(range(len(trainer.datamodule.train_set))) - set(preivous_indices))
            chosen_indices = np.random.choice(indices_pool, curr_query_budget, replace=False)
            new_indices = np.concatenate([preivous_indices, chosen_indices])
        else:
            train_state = pl_module.training
            pl_module.eval()
            logits = compute_logits_for_dataset(self.get_dataloader(trainer.datamodule), pl_module)

            pl_module.train(train_state)

            metric = unsupervised_margin(logits)
            metric = metric.detach().cpu().numpy()
            metric[preivous_indices] = np.inf
            chosen_indices = np.argsort(metric)[:curr_query_budget]
            new_indices = np.concatenate([preivous_indices, chosen_indices])

        trainer.datamodule.sampler.indices = new_indices

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.initialize_model()
        trainer.strategy.setup_optimizers(trainer)
        self.perform_extra_epochs(trainer, pl_module)
        # then we perform a single pytorch-lightning epoch...
