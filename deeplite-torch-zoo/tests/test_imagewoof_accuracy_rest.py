import pytest

from deeplite_torch_zoo import get_eval_function, get_model


@pytest.mark.parametrize(
    ('model_name', 'reference_accuracy'),
    [
        ('mobilenet_v3_small', 0.53125),
        ('squeezenet1_0', 0.46875),
        ('hrnet_w18_small_v2', 0.609375),
        ('efficientnet_es_pruned', 0.609375),
        ('mobilenetv2_w035', 0.375),
        ('mobileone_s0', 0.671875),
    ]
)