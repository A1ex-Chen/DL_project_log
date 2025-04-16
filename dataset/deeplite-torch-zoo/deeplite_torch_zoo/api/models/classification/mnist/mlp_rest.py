from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["mlp2_mnist", "mlp4_mnist", "mlp8_mnist"]

from deeplite_torch_zoo.src.classification.mnist_models.mlp import MLP

model_urls = {
    "mlp2": "http://download.deeplite.ai/zoo/models/mlp2-mnist-cd7538f979ca4d0e.pth",
    "mlp4": "http://download.deeplite.ai/zoo/models/mlp4-mnist-c6614ff040df60a4.pth",
    "mlp8": "http://download.deeplite.ai/zoo/models/mlp8-mnist-de6f135822553043.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='mlp2', dataset_name='mnist', task_type='classification'
)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='mlp4', dataset_name='mnist', task_type='classification'
)


@MODEL_WRAPPER_REGISTRY.register(
    model_name='mlp8', dataset_name='mnist', task_type='classification'
)