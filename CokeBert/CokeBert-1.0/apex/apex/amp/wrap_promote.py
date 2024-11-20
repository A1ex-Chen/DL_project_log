def promote(mod, fn, handle, verbose=False):
    orig_fn = utils.get_func(mod, fn)
    maybe_float = utils.verbosify(utils.maybe_float, fn, verbose)
    wrapper = make_promote_wrapper(orig_fn, maybe_float)
    utils.set_func_save(handle, mod, fn, wrapper)
