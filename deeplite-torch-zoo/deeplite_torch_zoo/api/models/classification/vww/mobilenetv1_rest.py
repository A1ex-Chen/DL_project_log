from deeplite_torch_zoo.src.classification.mobilenets.mobilenetv1 import MobileNetV1
from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY

__all__ = ["mobilenet_v1_vww", 'mobilenet_v1_025_vww', 'mobilenet_v1_025_96px_vww']

model_urls = {
    "mobilenet_v1": "http://download.deeplite.ai/zoo/models/mobilenetv1-vww-84f65dc4bc649cd6.pth",
    "mobilenet_v1_0.25": "http://download.deeplite.ai/zoo/models/mobilenet_v1_0.25.pt",
    "mobilenet_v1_0.25_96px": "http://download.deeplite.ai/zoo/models/mobilenet_v1_0.25_96px_798-8df2181bdab1433e.pt",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenet_v1', dataset_name='vww', task_type='classification'
)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenet_v1_0.25', dataset_name='vww', task_type='classification'
)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='mobilenet_v1_0.25_96px', dataset_name='vww', task_type='classification'
)