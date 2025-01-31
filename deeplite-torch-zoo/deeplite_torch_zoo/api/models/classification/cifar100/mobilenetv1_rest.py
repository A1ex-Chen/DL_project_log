"""mobilenet in pytorch

[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
[2] Implementation
    https://github.com/weiaicunzai/pytorch-cifar100

"""

from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.cifar_models.mobilenetv1 import MobileNet
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["mobilenet_v1_cifar100"]

model_urls = {
    "mobilenet_v1": "http://download.deeplite.ai/zoo/models/mobilenetv1-cifar100-4690c1a2246529eb.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenet_v1', dataset_name='cifar100', task_type='classification'
)