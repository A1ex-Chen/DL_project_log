# Code credit: https://github.com/Bobo-y/flexible-yolov5

import math

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from deeplite_torch_zoo.src.dnn_blocks.timm import DropBlock2D, LinearScheduler
from deeplite_torch_zoo.src.dnn_blocks.pytorchcv.cnn_attention import CBAM

from deeplite_torch_zoo.utils import LOGGER


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}






class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, cbam=False, dcn=False
    ):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        if not self.with_dcn:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        else:
            from torchvision.ops import DeformConv2d

            # deformable_groups = dcn.get('deformable_groups', 1)
            deformable_groups = 1
            offset_channels = 18
            self.conv2_offset = nn.Conv2d(
                planes, deformable_groups * offset_channels, kernel_size=3, padding=1
            )
            self.conv2 = DeformConv2d(
                planes, planes, kernel_size=3, padding=1, bias=False
            )
        self.bn2 = nn.BatchNorm2d(planes)
        self.use_cbam = cbam
        if self.use_cbam:
            self.cbam = CBAM(
                n_channels_in=self.expansion * planes, reduction_ratio=1, kernel_size=3
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)

        out = self.bn2(out)

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, cbam=False, dcn=False
    ):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if not self.with_dcn:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
        else:
            # deformable_groups = dcn.get('deformable_groups', 1)
            deformable_groups = 1
            from torchvision.ops import DeformConv2d

            offset_channels = 18
            self.conv2_offset = nn.Conv2d(
                planes,
                deformable_groups * offset_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
            )
            self.conv2 = DeformConv2d(
                planes, planes, kernel_size=3, padding=1, stride=stride, bias=False
            )

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_cbam = cbam
        if self.use_cbam:
            self.cbam = CBAM(
                n_channels_in=self.expansion * planes, reduction_ratio=1, kernel_size=3
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        cbam=False,
        dcn=False,
        drop_prob=0,
        base_channels=64,
        width=0.25,
        first_block_downsampling=False,
    ):
        super(Resnet, self).__init__()
        self.inplanes = int(base_channels * width)
        self.dcn = dcn
        self.cbam = cbam
        self.drop = drop_prob != 0
        self.conv1 = nn.Conv2d(
            3,
            int(base_channels * width),
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(int(base_channels * width))
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = []
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if self.drop:
            self.dropblock = LinearScheduler(
                DropBlock2D(drop_prob=drop_prob, block_size=5),
                start_value=0.0,
                stop_value=drop_prob,
                nr_steps=5e3,
            )
        ch = int(base_channels * width)
        self.layer1 = self._make_layer(
            block, ch, layers[0], stride=1 if not first_block_downsampling else 2
        )
        self.layer2 = self._make_layer(
            block, ch * 2, layers[1], stride=2, cbam=self.cbam
        )
        self.layer3 = self._make_layer(
            block, ch * 4, layers[2], stride=2, cbam=self.cbam, dcn=dcn
        )
        self.layer4 = self._make_layer(
            block, ch * 8, layers[3], stride=2, cbam=self.cbam, dcn=dcn
        )

        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

        self.out_shape = (
            self.out_channels[0] * 2,
            self.out_channels[1] * 2,
            self.out_channels[2] * 2,
        )

        LOGGER.info(
            "Backbone output channel: C3 {}, C4 {}, C5 {}".format(
                self.out_channels[0] * 2,
                self.out_channels[1] * 2,
                self.out_channels[2] * 2,
            )
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1, cbam=False, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, cbam=cbam, dcn=dcn)]
        self.inplanes = planes * block.expansion
        self.out_channels.append(self.inplanes)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cbam=cbam, dcn=dcn))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.drop:
            x1 = self.dropblock(self.layer1(x))
            x2 = self.dropblock(self.layer2(x1))
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x2, x3, x4















class Bottleneck(nn.Module):
    expansion = 4




class Resnet(nn.Module):





def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False
        )
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False
        )
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False
        )
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False
        )
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Resnet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False
        )
    return model


def resnet(pretrained=False, **kwargs):
    VERSION_FN_MAP = {
        '18': resnet18,
        '34': resnet34,
        '50': resnet50,
        '101': resnet101,
        '152': resnet152,
    }
    version = str(kwargs.pop('version'))
    return VERSION_FN_MAP[version](pretrained, **kwargs)