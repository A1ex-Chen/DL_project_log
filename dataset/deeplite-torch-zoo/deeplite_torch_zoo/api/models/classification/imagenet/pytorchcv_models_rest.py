from pytorchcv.model_provider import get_model as ptcv_get_model

from deeplite_torch_zoo.utils import load_state_dict_partial, LOGGER
from deeplite_torch_zoo.api.models.classification.model_implementation_dict import (
    MODEL_IMPLEMENTATIONS, PYTORCHCV_HAS_CHECKPOINT
)
from deeplite_torch_zoo.api.registries import MODEL_WRAPPER_REGISTRY
from deeplite_torch_zoo.api.models.classification.imagenet.utils import NUM_IMAGENET_CLASSES



    wrapper_func.__name__ = wrapper_fn_name
    return wrapper_func


for model_name_tag in MODEL_IMPLEMENTATIONS['pytorchcv']:
    register_model_name_tag = f'{model_name_tag}_pytorchcv'
    wrapper_name = '_'.join((model_name_tag, 'imagenet'))
    globals()[wrapper_name] = make_wrapper_func(
        wrapper_name,
        register_model_name_key=register_model_name_tag,
        model_name_key=model_name_tag,
    )