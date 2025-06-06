import pytest

from itertools import repeat
import torch.nn as nn

from deeplite_torch_zoo import get_model, get_zero_cost_estimator
from deeplite_torch_zoo.utils import init_seeds


REF_METRIC_VALUES = [
    ('fisher', 1.91525),
    ('grad_norm', 14.44768),
    ('grasp', -2.27878),
    ('jacob_cov', -32.233022),
    ('l2_norm', 172.978393),
    ('macs', 17794678784),
    ('nparams', 11220132),
    ('plain', 0.280363),
    ('snip', 317.93181),
    ('synflow', 3.31904e24),
    ('nwot_preact', 391.13919),
]


@pytest.mark.parametrize(
    ('metric_name', 'ref_value'),
    REF_METRIC_VALUES
)


@pytest.mark.parametrize(
    ('metric_name', 'ref_value'),
    REF_METRIC_VALUES
)

    loss = nn.CrossEntropyLoss()

    estimator_fn = get_zero_cost_estimator(metric_name=metric_name)
    metric_value = estimator_fn(model, model_output_generator=data_generator, loss_fn=loss_fn)
    assert pytest.approx(metric_value, rel=rel_tolerance) == ref_value