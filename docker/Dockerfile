FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.12-cuda11.3.1

RUN pip config unset global.cache-dir

RUN mkdir -p /opt/ml/wandb/.config

WORKDIR /opt/ml/code/
RUN pip install pip --upgrade
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
RUN pip install wandb
RUN pip install pytorch-lightning
RUN pip install timm
RUN pip install hydra-core --upgrade
RUN pip install numpy==1.21.0
RUN git clone -b v.0.2.8 https://github.com/VainF/Torch-Pruning /opt/ml/code/Torch-Pruning
ENV PYTHONPATH="$PYTHONPATH:/opt/ml/code/Torch-Pruning"
RUN pip install thop # for the pruning