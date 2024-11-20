def init():
    print('Initializing NVTX monkey patches')
    for cls in [torch, torch.Tensor, torch.nn.functional]:
        patchClass(cls)
    for cls in [torch.nn.RNN, torch.nn.RNNCell, torch.nn.LSTM, torch.nn.
        LSTMCell, torch.nn.GRU, torch.nn.GRUCell]:
        if isfunc(cls, 'forward'):
            add_wrapper(cls, 'forward')
    print('Done with NVTX monkey patching')
