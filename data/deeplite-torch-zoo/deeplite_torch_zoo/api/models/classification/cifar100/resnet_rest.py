"""resnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.resnet import (
    ResNet,
    BasicBlock,
    Bottleneck,
)
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = [
    "resnet18_cifar100",
    "resnet50_cifar100",
]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-cifar100-86b0c368c511bd57.pth",
    "resnet50": "http://download.deeplite.ai/zoo/models/resnet50-cifar100-d03f14e3031410de.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='resnet18', dataset_name='cifar100', task_type='classification'
)




@MODEL_WRAPPER_REGISTRY.register(
    model_name='resnet50', dataset_name='cifar100', task_type='classification'
)



