import torch
import torch.nn as nn
import torch.nn.functional as F
from darts.api import Model
from darts.genotypes import LINEAR_PRIMITIVES, Genotype
from darts.modules.classifier import MultitaskClassifier
from darts.modules.linear.cell import Cell


class Hyperparameters:
    c = 100
    num_nodes = 2
    num_cells = 3
    channel_multiplier = 1
    stem_channel_multiplier = 1
    intermediate_dim = 100


class LinearNetwork(Model):
    """Collection of cells"""








        gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self.num_nodes - self.channel_multiplier, self.num_nodes + 2)
        genotype = Genotype(
            normal=gene_normal,
            normal_concat=concat,
            reduce=gene_reduce,
            reduce_concat=concat,
        )

        return genotype