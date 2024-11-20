from . import compat
from . import utils
from ._amp_state import _amp_state
from . import rnn_compat

import functools

import torch



# `handle` arg is unused, but simplifies API to make `make_cast_wrapper`
# Annoyingly, make_promote_wrapper still uses the global handle.  Once everyone
# is on the new API and I am free to get rid of handle, I can clean this up.






# Current RNN approach:
# - Wrap top-level `RNN` function in thnn backend
# - Will call into either CudnnRNN or AutogradRNN
#  - Each of these are factory functions that return a per-iter
#    `forward` function
# - We interpose on the factory function to:
#   1) Interpose on the actual forward function and put in casts
#   2) Insert an fp16 `flat_weight` if necessary


    return wrapper

def cached_cast(mod, fn, cast_fn, handle,
                try_caching=False, verbose=False):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    cast_fn = utils.verbosify(cast_fn, fn, verbose)
    wrapper = make_cast_wrapper(orig_fn, cast_fn, handle, try_caching)
    utils.set_func_save(handle, mod, fn, wrapper)

# `handle` arg is unused, but simplifies API to make `make_cast_wrapper`
# Annoyingly, make_promote_wrapper still uses the global handle.  Once everyone
# is on the new API and I am free to get rid of handle, I can clean this up.
def make_promote_wrapper(orig_fn, cast_fn, handle=None):
    @functools.wraps(orig_fn)
    return wrapper

def promote(mod, fn, handle, verbose=False):
    orig_fn = utils.get_func(mod, fn)
    maybe_float = utils.verbosify(utils.maybe_float, fn, verbose)
    wrapper = make_promote_wrapper(orig_fn, maybe_float)
    utils.set_func_save(handle, mod, fn, wrapper)

def sequence_promote(mod, fn, handle, verbose=False):
    orig_fn = utils.get_func(mod, fn)
    maybe_float = utils.verbosify(utils.maybe_float, fn, verbose)
    @functools.wraps(orig_fn)
    utils.set_func_save(handle, mod, fn, wrapper)

def promote_match_arg0(mod, fn, handle, verbose=False):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    @functools.wraps(orig_fn)
    utils.set_func_save(handle, mod, fn, wrapper)

def err_if_any_half(mod, fn, handle, custom_err_msg=None):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    @functools.wraps(orig_fn)
    utils.set_func_save(handle, mod, fn, wrapper)

def err_if_arg0_half(mod, fn, handle, verbose=False):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    @functools.wraps(orig_fn)
    utils.set_func_save(handle, mod, fn, wrapper)

# Current RNN approach:
# - Wrap top-level `RNN` function in thnn backend
# - Will call into either CudnnRNN or AutogradRNN
#  - Each of these are factory functions that return a per-iter
#    `forward` function
# - We interpose on the factory function to:
#   1) Interpose on the actual forward function and put in casts
#   2) Insert an fp16 `flat_weight` if necessary
def rnn_cast(backend, fn, handle, verbose=False):
    orig_rnn = utils.get_func(backend, fn)
    @functools.wraps(orig_rnn)
    utils.set_func_save(handle, backend, fn, rnn_wrapper)

def new_rnn_cast(fn, handle, verbose=False):
    # Forward+backward compatibility around https://github.com/pytorch/pytorch/pull/15744
    # For rnn backend calls that route through _rnn_impls, we must patch the ref
    # that _rnn_impls stashed.  For rnn backend calls that directly invoke
    # _VF.<backend>, e.g. _VF.lstm, we can patch onto VariableFunctionsShim,
    # which in turn has patched the ref named "_VF" in torch.nn.modules.rnn.
    if utils.has_func(torch.nn.modules.rnn._rnn_impls, fn):
        mod = torch.nn.modules.rnn._rnn_impls
    else:
        mod = torch.nn.modules.rnn._VF
        assert isinstance(mod, rnn_compat.VariableFunctionsShim)
        fn = fn.lower()
    orig_fn = utils.get_func(mod, fn)
    cast_fn = utils.verbosify(utils.maybe_half, fn, verbose)
    @functools.wraps(orig_fn)
    utils.set_func_save(handle, mod, fn, wrapper)

def disable_casts(mod, fn, handle):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    @functools.wraps(orig_fn)
        return fwd_wrapper
    utils.set_func_save(handle, backend, fn, rnn_wrapper)

def new_rnn_cast(fn, handle, verbose=False):
    # Forward+backward compatibility around https://github.com/pytorch/pytorch/pull/15744
    # For rnn backend calls that route through _rnn_impls, we must patch the ref
    # that _rnn_impls stashed.  For rnn backend calls that directly invoke
    # _VF.<backend>, e.g. _VF.lstm, we can patch onto VariableFunctionsShim,
    # which in turn has patched the ref named "_VF" in torch.nn.modules.rnn.
    if utils.has_func(torch.nn.modules.rnn._rnn_impls, fn):
        mod = torch.nn.modules.rnn._rnn_impls
    else:
        mod = torch.nn.modules.rnn._VF
        assert isinstance(mod, rnn_compat.VariableFunctionsShim)
        fn = fn.lower()
    orig_fn = utils.get_func(mod, fn)
    cast_fn = utils.verbosify(utils.maybe_half, fn, verbose)
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        # Exact call signature from modules/rnn.py
        assert len(args) == 9
        assert len(kwargs) == 0

        if not _amp_state.handle.is_active():
            return orig_fn(*args, **kwargs)

        if isinstance(args[6], bool):
            params_idx = 2 # Not PackedSequence case
        else:
            params_idx = 3 # PackedSequence case

        new_args = []
        for i, arg in enumerate(args):
            if i == params_idx:
                num_params = sum([x.numel() for x in arg])
                fp16_weight_buf = args[0].new_empty((num_params,),
                                                    dtype=torch.half)
                casted_weights = utils.new_synthesize_flattened_rnn_weights(
                    arg, fp16_weight_buf, fn, verbose)
                new_args.append(casted_weights)
            elif utils.is_fp_tensor(arg):
                new_args.append(cast_fn(arg))
            else:
                new_args.append(arg)

        return orig_fn(*new_args)
    utils.set_func_save(handle, mod, fn, wrapper)

def disable_casts(mod, fn, handle):
    if not utils.has_func(mod, fn):
        return

    orig_fn = utils.get_func(mod, fn)
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        with handle._disable_casts():
            return orig_fn(*args, **kwargs)
    utils.set_func_save(handle, mod, fn, wrapper)