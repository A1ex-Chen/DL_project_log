"""dense net in pytorch

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.src.classification.cifar_models.densenet import (
    Bottleneck,
    DenseNet,
)
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY

__all__ = [
    "densenet121_cifar100",
]

model_urls = {
    "densenet121": "http://download.deeplite.ai/zoo/models/densenet121-cifar100-7e4ec64b17b04532.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='densenet121',
    dataset_name='cifar100',
    task_type='classification'
)





