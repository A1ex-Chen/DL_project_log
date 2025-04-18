import pytest
import torch

from deeplite_torch_zoo.src.dnn_blocks.effnet.effnet_blocks import FusedMBConv
from deeplite_torch_zoo.src.dnn_blocks.ghostnetv2.ghostnet_blocks import (
    GhostBottleneckV2, GhostModuleV2)
from deeplite_torch_zoo.src.dnn_blocks.mbnet.mbconv_blocks import MBConv
from deeplite_torch_zoo.src.dnn_blocks.mobileone.mobileone_blocks import (
    MobileOneBlock, MobileOneBlockUnit)
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.pelee_blocks import \
    TwoStackDenseBlock
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.shufflenet_blocks import \
    ShuffleUnit
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.squeezenet_blocks import (
    FireUnit, SqnxtUnit)
from deeplite_torch_zoo.src.dnn_blocks.replk.large_kernel_blocks import \
    RepLKBlock
from deeplite_torch_zoo.src.dnn_blocks.resnet.resnet_blocks import (
    ResNetBasicBlock, ResNetBottleneck, ResNeXtBottleneck)
from deeplite_torch_zoo.src.dnn_blocks.timm.regxnet_blocks import \
    RexNetBottleneck
from deeplite_torch_zoo.src.dnn_blocks.yolov7.repvgg_blocks import RepConv
from deeplite_torch_zoo.src.dnn_blocks.yolov7.transformer_blocks import (
    STCSPA, STCSPB, STCSPC, SwinTransformer2Block, SwinTransformerBlock,
    TransformerBlock)


@pytest.mark.parametrize(
    ('block', 'c1', 'c2', 'b', 'res', 'block_kwargs'),
    [
        (GhostModuleV2, 64, 64, 2, 32, {'dfc': False}),
        (GhostModuleV2, 64, 64, 2, 32, {'dfc': True}),
        (GhostBottleneckV2, 64, 64, 2, 32, {'mid_chs': 32}),
        (MobileOneBlock, 64, 64, 2, 32, {'use_se': True}),
        (MobileOneBlock, 64, 64, 2, 32, {'use_se': False}),
        (MobileOneBlockUnit, 64, 64, 2, 32, {'use_se': True}),
        (MobileOneBlockUnit, 64, 64, 2, 32, {'use_se': False}),
        (FireUnit, 64, 64, 2, 32, {}),
        (SqnxtUnit, 64, 64, 2, 32, {}),
        (RepConv, 64, 64, 2, 32, {}),
        (MBConv, 64, 64, 2, 32, {}),
        (FusedMBConv, 64, 64, 2, 32, {}),
        (TwoStackDenseBlock, 64, 64, 2, 32, {}),
        (RepLKBlock, 64, 64, 2, 32, {}),
        (RexNetBottleneck, 64, 64, 2, 32, {}),
        (ResNetBasicBlock, 64, 64, 2, 32, {}),
        (ResNetBottleneck, 64, 64, 2, 32, {}),
        (ResNeXtBottleneck, 64, 64, 2, 32, {}),
        (ShuffleUnit, 64, 64, 2, 32, {}),
        (TransformerBlock, 64, 64, 2, 32, {'num_heads': 2}),
        (SwinTransformerBlock, 64, 64, 2, 32, {'num_heads': 2}),
        (SwinTransformer2Block, 64, 64, 2, 32, {'num_heads': 2}),
        (STCSPA, 64, 64, 2, 32, {'transformer_block': TransformerBlock}),
        (STCSPB, 64, 64, 2, 32, {'transformer_block': TransformerBlock}),
        (STCSPC, 64, 64, 2, 32, {'transformer_block': TransformerBlock}),
    ],
)