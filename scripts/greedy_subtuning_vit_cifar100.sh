#!/bin/bash

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
    greedy_subset_selection=True \
    greedy_cross_val=True \
    greedy_eposilon=0.01 \
    greedy_start='all' \
    model.arch='vit_b_16' \
    model.learning_rate=0.0001 \
    model.weight_decay=0.00005 \
    datamodule.dataset=cifar100 \
    $OTHER_PARAMS
