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
    trainer.max_epochs=10 \
    mode='finetune_layers' \
    model.layers_to_finetune="[layer1_1]" \
    model.reinit_layers=True \
    $OTHER_PARAMS
