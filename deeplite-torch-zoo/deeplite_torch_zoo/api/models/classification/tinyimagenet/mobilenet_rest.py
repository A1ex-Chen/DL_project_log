from torchvision import models
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["mobilenet_v2_tinyimagenet"]


model_urls = {
    "mobilenet_v2": "http://download.deeplite.ai/zoo/models/mobilenet_v2_tinyimagenet_0-6803-4ec21929b72f0b4d.pt",
}


@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenet_v2', dataset_name='tinyimagenet', task_type='classification'
)