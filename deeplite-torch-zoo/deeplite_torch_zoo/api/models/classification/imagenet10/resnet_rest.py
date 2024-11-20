import torchvision
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["resnet18_imagenet10"]

model_urls = {
    "resnet18": "http://download.deeplite.ai/zoo/models/resnet18-imagenet10-f119488aa5e047b0.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='resnet18', dataset_name='imagenet10', task_type='classification'
)