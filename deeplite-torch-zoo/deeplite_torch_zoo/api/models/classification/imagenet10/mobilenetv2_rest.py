import torchvision
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["mobilenet_v2_0_35_imagenet10"]

model_urls = {
    "mobilenetv2_0.35": "http://download.deeplite.ai/zoo/models/mobilenetv2_0.35-imagenet10-2410796e32dbde1c.pth",
    "mobilenetv2_1.0": "",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenet_v2_0_35',
    dataset_name='imagenet10',
    task_type='classification',
)

