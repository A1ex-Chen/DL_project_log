from typing import Any, List
import string
from xmlrpc.client import Boolean
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
# from pytorch_lightning import LightningModule

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_vit,
)
import sys


class DepthLoss(nn.Module):



class DPT(BaseModel):
                



class DPTDepthModel(DPT):
        
        # if freeze:
        #     for name, p in self.named_parameters():
        #         print(name)
        #         p.requires_grad = False




# class LitDPTModule(LightningModule):

#     def __init__(
#         self,
#         path: string = None,
#         non_negative: bool = True,
#         scale: float = 0.000305,
#         shift: float = 0.1378,
#         invert: bool = False,
#         lr: float = 0.0001,
#         weight_decay: float = 0.005,
#         loss_type: string = "eigen",
#     ):
#         super().__init__()

#         # this line allows to access init params with 'self.hparams' attribute
#         # it also ensures init params will be stored in ckpt
#         self.save_hyperparameters(logger=False)

#         self.model = DPTDepthModel(path, non_negative, scale, shift, invert)

#         # loss function
#         self.criterion = DepthLoss(loss_type)

#         # self.automatic_optimization = False
    
#     def forward(self, x: torch.Tensor):
#         return self.model(x)

#     def step(self, batch: Any):
#         in_, mask, gt = batch['image'], batch['mask'], batch['depth']
#         pred = self.forward(in_)
#         loss = self.criterion(pred, gt)
#         return loss, pred, gt

#     def training_step(self, batch: Any, batch_idx: int):
        
#         # opt = self.optimizers()
#         # opt.zero_grad()

#         loss, preds, targets = self.step(batch)
#         self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
#         # input_visual = rgb_unnormalize(batch['image'][0])
#         # preds_visual = depth_visualization(preds[0])
#         # gt_visual = depth_visualization(batch['depth'][0])
#         # tensor_logger = self.logger.experiment[0]
#         # tensor_logger.add_image(
#         #     'train/input_rgb', input_visual, self.global_step
#         # )
#         # tensor_logger.add_image(
#         #     'train/pred_depth', preds_visual, self.global_step
#         # )
#         # tensor_logger.add_image(
#         #     'train/gt_depth', gt_visual, self.global_step
#         # )

#         # we can return here dict with any tensors
#         # and then read it in some callback or in `training_epoch_end()`` below
#         # remember to always return loss from `training_step()` or else backpropagation will fail!
#         # self.manual_backward(loss)

#         return loss
        
#     def training_epoch_end(self, outputs: List[Any]):
#         # `outputs` is a list of dicts returned from `training_step()`
#         pass

#     def validation_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)
#         self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
#         input_visual = rgb_unnormalize(batch['image'][0])
#         preds_visual = depth_visualization(preds[0])
#         gt_visual = depth_visualization(batch['depth'][0])
#         tensor_logger = self.logger.experiment[0]
#         tensor_logger.add_image(
#             f'val/input_rgb', input_visual, self.global_step
#         )
#         tensor_logger.add_image(
#             f'val/pred_depth', preds_visual, self.global_step
#         )
#         tensor_logger.add_image(
#             f'val/gt_depth', gt_visual, self.global_step
#         )

#         return loss

#     def validation_epoch_end(self, outputs: List[Any]):
#         # acc = self.val_acc.compute()  # get val accuracy from current epoch
#         # self.val_acc_best.update(acc)
#         # self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
#         pass

#     def test_step(self, batch: Any, batch_idx: int):
#         loss, preds, targets = self.step(batch)
#         self.log("test/loss", loss, on_step=False, on_epoch=True)

#         return loss

#     def test_epoch_end(self, outputs: List[Any]):
#         pass

#     def on_epoch_end(self):
#         # reset metrics at the end of every epoch
#         pass

#     def configure_optimizers(self):
#         """Choose what optimizers and learning-rate schedulers to use in your optimization.
#         Normally you'd need one. But in the case of GANs or similar you might have multiple.
#         See examples here:
#             https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
#         """
#         return torch.optim.Adam(
#             params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
#         )