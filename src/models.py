import math
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax
from torchvision import models
from torchvision.models import ResNet50_Weights, vit_b_16

from src.model_utils import SelectiveFinetuneResNet, SelectiveFinetuneViT
import wandb
from src.datamodules import SUPPORTED_DATASETS_CLASSES


def get_num_classes(dataset):
    if dataset.lower() in SUPPORTED_DATASETS_CLASSES:
        return SUPPORTED_DATASETS_CLASSES[dataset.lower()]
    else:
        raise ValueError(f"dataset must be in {SUPPORTED_DATASETS_CLASSES.keys()}")


MODEL_LAYERS = {
    "resnet50": [
        "layer1_0", "layer1_1", "layer1_2", "layer2_0", "layer2_1", "layer2_2", "layer2_3",
        "layer3_0", "layer3_1", "layer3_2", "layer3_3", "layer3_4", "layer3_5", "layer4_0",
        "layer4_1", "layer4_2"
    ],
    "resnet18": [
        "layer1_0", "layer1_1", "layer2_0"
        "layer2_1", "layer3_0", "layer3_1", "layer4_0", "layer4_1"
    ],
    "vit_b_16": [
        "encoder_layer_0", "encoder_layer_1", "encoder_layer_2", "encoder_layer_3",
        "encoder_layer_4", "encoder_layer_5", "encoder_layer_6", "encoder_layer_7",
        "encoder_layer_8", "encoder_layer_9", "encoder_layer_10", "encoder_layer_11"
    ],
}


def get_head_params(model_name, backbone, num_classes, head_depth, head_width, sidenet_level):
    in_features = backbone.fc.in_features if "resnet" in model_name.lower(
    ) else backbone.heads.head.in_features  # Assumes the possible models are resnet or vit

    head_params = {
        "in_features": in_features,
        "num_classes": num_classes,
        "depth": head_depth,
        "width": head_width,
    }
    return head_params


def get_model(model_name,
              num_classes,
              mode,
              head_width,
              head_depth,
              sidenet_level=3,
              use_frozen_bb=True,
              layers_to_finetune=None,
              reinit_layers=False):

    if model_name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        backbone = models.resnet50(num_classes=1000, weights=weights)
        model_class = SelectiveFinetuneResNet
        model_layers = MODEL_LAYERS["resnet50"]
    elif model_name == "resnet50dino":
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        # hack for code consistency
        backbone.fc = nn.Linear(2048, 1000)
        model_class = SelectiveFinetuneResNet
        model_layers = MODEL_LAYERS["resnet50"]
    elif model_name == "resnet18":
        backbone = models.resnet18(pretrained=True)
        model_class = SelectiveFinetuneResNet
        model_layers = MODEL_LAYERS["resnet18"]
    elif model_name == "vit_b_16":
        backbone = vit_b_16(pretrained=True)
        model_class = SelectiveFinetuneViT
        model_layers = MODEL_LAYERS["vit_b_16"]
    else:
        raise ValueError(f"model_name must be in ['resnet50', 'resnet18'], but got {model_name}")

    head_params = get_head_params(model_name, backbone, num_classes, head_depth, head_width,
                                  sidenet_level)

    if mode == 'linear':
        layers_to_finetune = []
        model = model_class(backbone,
                            head_params=head_params,
                            layers_to_finetune=layers_to_finetune,
                            use_freeze_bb=use_frozen_bb,
                            reinit_layers=reinit_layers)
    elif mode == 'finetune':
        layers_to_finetune = model_layers
        model = model_class(backbone,
                            head_params=head_params,
                            layers_to_finetune=layers_to_finetune,
                            use_freeze_bb=use_frozen_bb,
                            reinit_layers=reinit_layers)
    elif mode == 'finetune_layers':
        model = model_class(backbone,
                            head_params=head_params,
                            layers_to_finetune=layers_to_finetune,
                            use_freeze_bb=use_frozen_bb,
                            reinit_layers=reinit_layers)
    else:
        raise ValueError(
            f"mode must be in ['linear', 'finetune', 'finetune_layers'], but got {mode}")

    return model


class LitModel(pl.LightningModule):

    def __init__(self,
                 arch: str = "resnet50",
                 learning_rate: float = 1e-1,
                 weight_decay: float = 1e-4,
                 max_epochs: int = 50,
                 schedule: str = 'step',
                 dataset: str = 'cifar10',
                 mode: str = False,
                 head_width: int = 64,
                 head_depth: int = 1,
                 optimizer: str = "sgd",
                 sidenet_level: int = 3,
                 use_frozen_bb: bool = True,
                 layers_to_finetune: list = None,
                 reinit_layers: bool = False):
        super().__init__()
        assert mode in ['linear', 'finetune', 'finetune_layers']

        self.criterion = torch.nn.CrossEntropyLoss()
        print(arch)
        self.model_parameters = dict(
            model_name=arch,
            num_classes=get_num_classes(dataset),
            mode=mode,
            head_width=head_width,
            head_depth=head_depth,
            sidenet_level=sidenet_level,
            use_frozen_bb=use_frozen_bb,
            layers_to_finetune=layers_to_finetune,
            reinit_layers=reinit_layers,
        )
        self.initialize_model()
        with open('model.txt', 'w') as fp:
            print(self.model, file=fp)
        self.train_acc = Accuracy()
        self.pred_accs = Accuracy()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.opt_max_epochs = max_epochs

        self.schedule = schedule

        self.save_hyperparameters()  # for model checkpointing
        self.optimizer_name = optimizer

    def initialize_model(self):
        self.model = get_model(**self.model_parameters)
        self.model = self.model.to(self.device)
        #move model to channels last
        self.model = self.model.to(memory_format=torch.channels_last)

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        images, _ = batch
        return self.model(images)

    def process_batch(self, batch, stage="train"):
        images, labels = batch
        logits = self.forward(images)
        probs = softmax(logits, dim=1)
        loss = self.criterion(logits, labels)

        if stage == "train":
            self.train_acc(probs, labels)
        elif stage == "pred":
            self.pred_accs(probs, labels)
        else:
            raise ValueError("Invalid stage %s" % stage)

        return loss

    def training_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "train")
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "pred")
        self.log("pred_loss", loss)
        self.log("pred_acc", self.pred_accs, on_epoch=True)

    def configure_optimizers(self):
        parameters = self.model.parameters()
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(parameters,
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay,
                                        momentum=0.9)

        elif self.optimizer_name == "adamw":
            print('using adamw')
            optimizer = torch.optim.AdamW(parameters,
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters,
                                         lr=self.learning_rate,
                                         weight_decay=self.weight_decay)
        if self.schedule == 'step':
            print('using step lr scheduler')
            lr_scheduler = lr_sched.StepLR(optimizer, step_size=self.max_epochs // 3 + 1, gamma=0.1)
        elif self.schedule == 'cos':
            t_max = self.opt_max_epochs
            print('using cosine lr scheduler with t_max =', t_max)
            lr_scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=t_max)
        elif self.schedule == 'const':
            print('using constant lr scheduler')
            lr_scheduler = lr_sched.ConstantLR(optimizer, factor=1)
        else:
            raise NotImplementedError()

        return [optimizer], [lr_scheduler]

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        return callbacks
