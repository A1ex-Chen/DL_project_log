# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch.nn as nn

from src.models.model_utils import SqueezeAndExcitation ,scSE,ChSqueezeAndSpExcitation,SpSqueezeAndChExcitation,ESqueezeAndExcitation,SPALayer,SPABLayer,SPACLayer


class SqueezeAndExciteFusionAdd(nn.Module):
        

    
class ESqueezeAndExciteFusionAdd(nn.Module):
