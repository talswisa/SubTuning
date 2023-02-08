#!/bin/bash

export HYDRA_FULL_ERROR=1

export WANDB_ENTITY=$(grep 'WANDB_ENTITY' wandb.text | cut -c 15-)
export WANDB_API_KEY=$(grep 'WANDB_API_KEY' wandb.text | cut -c 16-)
WANDB_PROJECT=$(grep 'WANDB_PROJECT' wandb.text | cut -c 16-)

echo WANDB_ENTITY=$WANDB_ENTITY
echo WANDB_PROJECT=$WANDB_PROJECT

OTHER_PARAMS=${@:1}

python ./run.py -m \
    logger.project=$WANDB_PROJECT \
    trainer.max_epochs=6 \
    trainer.reload_dataloaders_every_n_epochs=1 \
    model.arch='resnet50' \
    model.layers_to_finetune="[layer2_3]" \
    mode='finetune_layers' \
    datamodule.do_active_learning=true \
    active_learning.n_epochs=50 \
    active_learning.metric='unsupervised_margin' \
    active_learning.query_budget="[100, 400, 500, 1500, 2500, 5000, 10000]" \
    $OTHER_PARAMS

# trainer.max_epochs=6: number of active learning iterations
# active_learning.mode: 'finetune_layers' for subtuning, or 'finetune'
# active_learning.n_epochs: number of epochs for each active learning iteration
# active_learning.layers_to_finetune: which layers to train
# active_learning.query_budget: number of samples to query at each active learning iteration  (the last values is not important, we will not train after the last query)

