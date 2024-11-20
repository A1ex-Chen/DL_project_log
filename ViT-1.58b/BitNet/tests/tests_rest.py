import pytest
import torch
from torch.nn import functional as F

from bitnet.bitlinear import BitLinear, absmax_quantize
from bitnet.bit_transformer import (
    BitNetTransformer,
    ParallelTransformerBlock,
    Transformer,
)

# Basic Tests:








# Fixtures:


@pytest.fixture


# Parameterized Testing:


@pytest.mark.parametrize("bits", [4, 8, 12, 16])


# More Tests for BitLinear:


@pytest.mark.parametrize(
    "in_features,out_features", [(10, 20), (20, 40), (5, 10), (15, 10)]
)


@pytest.mark.parametrize("groups", [1, 2, 5])




@pytest.mark.parametrize("groups", [1, 2, 5])






@pytest.mark.parametrize("groups", [1, 2, 5])






# ... Continue adding more tests ...
# - Test the forward pass with extreme input values.
# - Test with different types of input tensors (e.g., int8, float16).
# - Test the forward pass with batch sizes other than 5.
# - Verify that using different initializations produces different results.
# - Test the weight and input interactions during the forward pass.
# - And many more...

# ================================ Transformer with bitlinear ================================


@pytest.fixture


@pytest.fixture


@pytest.mark.parametrize(
    "dim, dim_head, heads, ff_mult",
    [
        (512, 64, 8, 4),
        (256, 32, 4, 2),
        (128, 16, 2, 1),
    ],
)


@pytest.mark.parametrize(
    "dim, depth, heads, dim_head, ff_mult",
    [
        (512, 6, 8, 64, 4),
        (256, 3, 4, 32, 2),
        (128, 2, 2, 16, 1),
    ],
)








@pytest.mark.parametrize(
    "dim, dim_head, heads, ff_mult",
    [
        (512, 64, 8, 4),
        (256, 32, 4, 2),
        (128, 16, 2, 1),
    ],
)


@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (1, 512),
        (32, 128),
        (64, 256),
    ],
)




@pytest.mark.parametrize("mask_value", [100, 200, 300])


@pytest.mark.parametrize(
    "input_value, expected_output_shape",
    [
        (torch.randint(0, 20000, (1, 512)), (1, 20000)),
        (torch.randint(0, 20000, (32, 256)), (32, 20000)),
    ],
)






# Mocking and Monkeypatching




# Add more tests based on the scenarios and edge cases you want to cover.