"""mobilenetv2 in pytorch

[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""


from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.mobilenetv2 import MobileNetV2
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["mobilenet_v2_cifar100"]

model_urls = {
    "mobilenet_v2": "http://download.deeplite.ai/zoo/models/mobilenetv2-cifar100-a7ba34049d626cf4.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenet_v2', dataset_name='cifar100', task_type='classification'
)