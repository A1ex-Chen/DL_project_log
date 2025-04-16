import torchvision
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["resnet18_imagenet16", "resnet50_imagenet16"]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-imagenet16-2f8c56bafc30cde9.pth",
    "resnet50": "http://download.deeplite.ai/zoo/models/resnet50-imagenet16-f546a9fdf7bff1b9.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='resnet18', dataset_name='imagenet16', task_type='classification'
)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='resnet50', dataset_name='imagenet16', task_type='classification'
)