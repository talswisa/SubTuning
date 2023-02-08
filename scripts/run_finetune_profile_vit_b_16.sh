#!/bin/bash

export HYDRA_FULL_ERROR=1

export WANDB_ENTITY=$(grep 'WANDB_ENTITY' wandb.text | cut -c 15-)
export WANDB_API_KEY=$(grep 'WANDB_API_KEY' wandb.text | cut -c 16-)
WANDB_PROJECT=$(grep 'WANDB_PROJECT' wandb.text | cut -c 16-)

echo WANDB_ENTITY=$WANDB_ENTITY
echo WANDB_PROJECT=$WANDB_PROJECT

OTHER_PARAMS=${@:1}

layers=(encoder_layer_0 encoder_layer_1 encoder_layer_2 encoder_layer_3 encoder_layer_4 encoder_layer_5 encoder_layer_6 encoder_layer_7 encoder_layer_8 encoder_layer_9 encoder_layer_10 encoder_layer_11)

for layer in "${layers[@]}"; do
    echo "Finetuning $layer"
    python ./run.py -m \
        logger.project=$WANDB_PROJECT \
        trainer.max_epochs=10 \
        trainer.precision=32 \
        mode='finetune_layers' \
        model.layers_to_finetune="[$layer]" \
        model.arch='vit_b_16' \
        model.learning_rate=0.0001 \
        model.weight_decay=0.00005 \
        $OTHER_PARAMS
done
