from typing import Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from arch_style_ml.base import ModelBase
from arch_style_ml.utils import setup_logger

import cytoolz as toolz


log = setup_logger(__name__)

MODELS = {
    "inception_v3": lambda *args: models.inception_v3(
        pretrained=True, aux_logits=False
    ),
    "resnext": lambda *args: models.resnext50_32x4d(pretrained=True, progress=True),
}


class ArchStyleModel(ModelBase):
    def __init__(
        self,
        base_model: Union[str, Callable[[None,], nn.Module]] = MODELS["resnext"],
        fc_layer: nn.Module = nn.Linear,
        n_classes: int = 25,
    ):
        super().__init__()
        if isinstance(base_model, str):
            base_model = MODELS.get(base_model, MODELS["resnext"])
        self.base_model = base_model()
        for param in self.base_model.parameters():
            param.requires_grad = False

        in_fts = self.base_model.fc.in_features
        self.base_model.fc = fc_layer(in_fts, n_classes)

    def forward(self, *args, **kwargs):
        return self.base_model.forward(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return toolz.pipe(
            self.forward(*args, **kwargs),
            F.softmax(self.base_model.fc.in_features, self.base_model.fc.out_features),
        )

    def predict(self, *args, **kwargs):
        return toolz.pipe(self.forward(*args, **kwargs), torch.max)
