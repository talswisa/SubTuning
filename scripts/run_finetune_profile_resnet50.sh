#!/bin/bash

export HYDRA_FULL_ERROR=1

export WANDB_ENTITY=$(grep 'WANDB_ENTITY' wandb.text | cut -c 15-)
export WANDB_API_KEY=$(grep 'WANDB_API_KEY' wandb.text | cut -c 16-)
WANDB_PROJECT=$(grep 'WANDB_PROJECT' wandb.text | cut -c 16-)

echo WANDB_ENTITY=$WANDB_ENTITY
echo WANDB_PROJECT=$WANDB_PROJECT

OTHER_PARAMS=${@:1}

layers=(layer1_0 layer1_1 layer1_2 layer2_0 layer2_1 layer2_2 layer2_3 layer3_0 layer3_1 layer3_2 layer3_3 layer3_4 layer3_5 layer4_0 layer4_1 layer4_2)

for layer in "${layers[@]}"; do
    echo "Finetuning $layer"
    python ./run.py -m \
        logger.project=$WANDB_PROJECT \
        trainer.max_epochs=10 \
        mode='finetune_layers' \
        model.layers_to_finetune="[$layer]" \
        $OTHER_PARAMS
done
