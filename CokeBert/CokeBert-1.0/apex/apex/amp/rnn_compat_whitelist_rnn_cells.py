def whitelist_rnn_cells(handle, verbose):
    if has_old_rnns():
        fn_names = ['RNNReLUCell', 'RNNTanhCell', 'LSTMCell', 'GRUCell']
        mod = torch.nn.backends.thnn.backend
    else:
        fn_names = [(x + '_cell') for x in RNN_NAMES]
        mod = torch.nn.modules.rnn._VF
        assert isinstance(mod, VariableFunctionsShim)
    for fn in fn_names:
        wrap.cached_cast(mod, fn, utils.maybe_half, handle, try_caching=
            True, verbose=verbose)
    if has_old_rnns():
        for rnn_type in ['GRUFused', 'LSTMFused']:
            mod = getattr(torch.nn._functions.thnn.rnnFusedPointwise, rnn_type)
            wrap.disable_casts(mod, 'backward', handle)
