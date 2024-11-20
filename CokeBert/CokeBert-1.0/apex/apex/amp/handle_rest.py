import contextlib
import warnings
import torch

from . import utils
from .opt import OptimWrapper
from .scaler import LossScaler
from ._amp_state import _amp_state, master_params, maybe_print
from ..fp16_utils import FP16_Optimizer as FP16_Optimizer_general
from ..optimizers import FP16_Optimizer as FP16_Optimizer_for_fused
from ..parallel.LARC import LARC


# There's no reason to expose the notion of a "handle". Everything can happen through amp.* calls.
@contextlib.contextmanager


# Free function version of AmpHandle.disable_casts, another step on the
# path to removing the concept of "AmpHandle"
@contextlib.contextmanager


class AmpHandle(object):


    @contextlib.contextmanager


    @contextlib.contextmanager


    # Experimental support for saving / restoring uncasted versions of functions


    @property

    @property


    @property

class NoOpHandle(object):

    @contextlib.contextmanager


    @contextlib.contextmanager

    @property

    @property


                            return skip_step
                        optimizer.step = patch_step(optimizer, loss_scaler, loss_id)
                        optimizer._amp_stash.already_patched = True

    # Probably ok to skip this if not delay_unscale
    if _amp_state.opt_properties.patch_torch_functions:
        _amp_state.handle._clear_cache()


# Free function version of AmpHandle.disable_casts, another step on the
# path to removing the concept of "AmpHandle"
@contextlib.contextmanager
def disable_casts():
    _amp_state.handle._is_active = False
    yield
    _amp_state.handle._is_active = True


class AmpHandle(object):
    def __init__(self, loss_scale="dynamic", enable_caching=True, verbose=False):
        self._enable_caching = enable_caching
        self._verbose = verbose
        self._cache = dict()
        self._default_scaler = LossScaler(loss_scale)
        self._is_active = True
        self._all_wrappers = []

    def is_active(self):
        return self._is_active

    @contextlib.contextmanager
    def _disable_casts(self):
        self._is_active = False
        yield
        self._is_active = True

    def wrap_optimizer(self, optimizer, num_loss=1):
        self._default_scaler = None
        return OptimWrapper(optimizer, self, num_loss)

    @contextlib.contextmanager
    def scale_loss(self, loss, optimizer):
        raise RuntimeError("The old Amp API is no longer supported.  Please move to the new API, "
            "documented here:  https://nvidia.github.io/apex/amp.html.  Transition guide:  "
            "https://nvidia.github.io/apex/amp.html#transition-guide-for-old-api-users")

        if not self.is_active():
            yield loss
            return

        if self._default_scaler is None:
            raise RuntimeError(
                'After calling `handle.wrap_optimizer()`, you must explicitly ' +
                'use `optimizer.scale_loss(loss)`.')

        # TODO: this code block is duplicated here and `opt.py`. Unify.
        loss_scale = self._default_scaler.loss_scale()
        yield loss * loss_scale

        self._default_scaler.clear_overflow_state()
        self._default_scaler.unscale(
            master_params(optimizer),
            master_params(optimizer),
            loss_scale)
        should_skip = self._default_scaler.update_scale()
        if should_skip:
            optimizer_step = optimizer.step
            def skip_step():
                maybe_print('Gradient overflow, skipping update')
                optimizer.step = optimizer_step
            optimizer.step = skip_step

        self._clear_cache()

    def _clear_cache(self):
        self._cache.clear()

    # Experimental support for saving / restoring uncasted versions of functions
    def _save_func(self, mod, fn, func):
        self._all_wrappers.append((mod, fn, func))

    def _deactivate(self):
        for mod, fn, func in self._all_wrappers:
            utils.set_func(mod, fn, func)
        self._all_wrappers = []

    @property
    def has_cache(self):
        return self._enable_caching

    @property
    def cache(self):
        return self._cache

    def remove_cache(self, param):
        if self.has_cache and param in self.cache:
            del self.cache[param]

    @property
    def verbose(self):
        return self._verbose

class NoOpHandle(object):
    def is_active(self):
        return False

    @contextlib.contextmanager
    def _disable_casts(self):
        yield

    def wrap_optimizer(self, optimizer, num_loss=1):
        return OptimWrapper(optimizer, self, num_loss)

    @contextlib.contextmanager
    def scale_loss(self, loss, optimizer):
        yield loss

    @property
    def has_cache(self):
        return False

    @property
    def verbose(self):
        return False

    def _clear_cache(self):
        pass

    def _deactivate(self):
        pass