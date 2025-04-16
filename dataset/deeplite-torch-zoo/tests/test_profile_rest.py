import pytest

from deeplite_torch_zoo import get_model, profile


@pytest.mark.parametrize(
    ('ref_model_name', 'ref_gmacs', 'ref_mparams', 'ref_peakram'),
    [
        ('resnet50', 4.100300288, 25.557032, 9.1875)
    ]
)