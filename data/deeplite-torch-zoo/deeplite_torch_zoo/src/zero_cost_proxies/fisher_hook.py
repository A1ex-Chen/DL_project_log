def hook(module, grad_input, grad_output):
    act = layer.act.detach()
    grad = grad_output[0].detach()
    if len(act.shape) > 2:
        g_nk = torch.sum(act * grad, list(range(2, len(act.shape))))
    else:
        g_nk = act * grad
    del_k = g_nk.pow(2).mean(0).mul(0.5)
    if layer.fisher is None:
        layer.fisher = del_k
    else:
        layer.fisher += del_k
    del layer.act
