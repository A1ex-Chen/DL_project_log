"""preactresnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""


from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.preact_resnet import (
    PreActResNet,
    PreActBlock,
    PreActBottleneck,
)
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["pre_act_resnet18_cifar100"]

model_urls = {
    "pre_act_resnet18": "http://download.deeplite.ai/zoo/models/pre_act_resnet18-cifar100-1c4d1dc76ee9c6f6.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='pre_act_resnet18', dataset_name='cifar100', task_type='classification'
)







