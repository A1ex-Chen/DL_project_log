import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from models.pointbert.dvae import Group
from models.pointbert.dvae import Encoder
from models.pointbert.logger import print_log

from models.pointbert.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

class Mlp(nn.Module):



class Attention(nn.Module):



class Block(nn.Module):



class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """




class PointTransformer(nn.Module):
        # self.load_model_from_ckpt('/export/home/repos/SLIP/pretrained_models/point_transformer_8192.pt')
        # if not self.args.evaluate_3d:
        #     self.load_model_from_ckpt('./data/initialize_models/point_bert_pretrained.pt')

        # self.cls_head_finetune = nn.Sequential(
        #     nn.Linear(self.trans_dim * 2, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, self.cls_dim)
        # )

        # self.build_loss_func()



