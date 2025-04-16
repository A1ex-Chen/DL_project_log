"""

[1] Implementation
    https://github.com/kuangliu/pytorch-cifar

"""

from deeplite_torch_zoo.utils import load_pretrained_weights
from deeplite_torch_zoo.src.classification.mnist_models.lenet import LeNet5
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY


__all__ = ["lenet5_mnist"]

model_urls = {
    "lenet5": "http://download.deeplite.ai/zoo/models/lenet-mnist-e5e2d99e08460491.pth",
}




@MODEL_WRAPPER_REGISTRY.register(
    model_name='lenet5', dataset_name='mnist', task_type='classification'
)