import math
from peft.tuners.lora import LoraLayer
from peft.utils import transpose
import torch
import torch.nn as nn
import torch.nn.functional as F


class Top2Gating(nn.Module):
    MIN_EXPERT_CAPACITY = 4
    

    @staticmethod

    

class MoeLoraLayer(LoraLayer):

        




class MoeLinear(nn.Linear, MoeLoraLayer):


    
    
    