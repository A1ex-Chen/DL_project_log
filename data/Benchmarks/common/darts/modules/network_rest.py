from typing import Callable, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.api import Model
from darts.genotypes import Genotype
from darts.modules import Cell
from darts.modules.classifier import MultitaskClassifier


class Hyperparameters:
    c = 1
    num_nodes = 2
    num_cells = 3
    channel_multiplier = 1


class Network(Model):
    """Collection of cells

    Args:
        stem: nn.Module that takes the input data
              and outputs `cell_dim` number of features

        classifier_dim: number of features from
              Darts.modules.mixed_layer.MixedLayer. This
              depends upon the choice of primitives specified
              by `ops`.

        ops: Constructor for all of the primitive nn.Modules. This
             should be a dictionary of lambda function used to construct
             your nn.Modules. The parameters of the lamdas must be `c`, the
             number of input channels of each primitive, `stride`, the stride for
             convolution blocks, and `affine`, whether to use `affine` in
             batch norm.

        tasks: a dictionary whose keys are the names of the classification
               tasks, and whose keys are the number of classes in each task.

        criterion: Pytorch loss criterion

        device: Either "cpu" or "gpu

        hyperparams: instance of Hyperparameters. This hyperparamters for DARTS.
    """







        gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        concat = range(2 + self.num_nodes - self.channel_multiplier, self.num_nodes + 2)

        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_normal,
            reduce_concat=concat,
        )

        return genotype