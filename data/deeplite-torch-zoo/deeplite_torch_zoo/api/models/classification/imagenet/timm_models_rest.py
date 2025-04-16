import timm

from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.api.models.classification.imagenet.utils import NUM_IMAGENET_CLASSES



    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in timm.list_models():
    wrapper_name = '_'.join((model_name_tag, 'imagenet'))
    globals()[wrapper_name] = make_wrapper_func(wrapper_name, model_name_tag)