import inspect

from diffusers.utils import is_flax_available
from diffusers.utils.testing_utils import require_flax


if is_flax_available():
    import jax


@require_flax
class FlaxModelTesterMixin:

