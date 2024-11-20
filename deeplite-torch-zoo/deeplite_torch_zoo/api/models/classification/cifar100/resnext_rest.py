"""resnext in pytorch

[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He.
    Aggregated Residual Transformations for Deep Neural Networks
    https://arxiv.org/abs/1611.05431
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.resnext import ResNeXt
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["resnext29_2x64d_cifar100"]

model_urls = {
    "resnext29_2x64d": "http://download.deeplite.ai/zoo/models/resnext29_2x64d-cifar100-f6ba33baf30048d1.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='resnext29_2x64d', dataset_name='cifar100', task_type='classification'
)





