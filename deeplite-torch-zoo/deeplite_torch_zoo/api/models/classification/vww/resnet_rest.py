import torchvision

from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["resnet18_vww", "resnet50_vww"]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-vww-7f02ab4b50481ab7.pth",
    "resnet50": "http://download.deeplite.ai/zoo/models/resnet50-vww-9d4cb2cb19f8c5d5.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='resnet18', dataset_name='vww', task_type='classification'
)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='resnet50', dataset_name='vww', task_type='classification'
)