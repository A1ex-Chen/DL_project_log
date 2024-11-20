import unittest

from diffusers import FlaxAutoencoderKL
from diffusers.utils import is_flax_available
from diffusers.utils.testing_utils import require_flax

from ..test_modeling_common_flax import FlaxModelTesterMixin


if is_flax_available():
    import jax


@require_flax
class FlaxAutoencoderKLTests(FlaxModelTesterMixin, unittest.TestCase):
    model_class = FlaxAutoencoderKL

    @property
