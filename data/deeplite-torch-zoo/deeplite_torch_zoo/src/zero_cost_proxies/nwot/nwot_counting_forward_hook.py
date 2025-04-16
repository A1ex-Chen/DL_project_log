def counting_forward_hook(module, inp, out):
    if pre_act:
        out = out.view(out.size(0), -1)
        x = (out > 0).float()
    else:
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.view(inp.size(0), -1)
        x = (inp > 0).float()
    K = x @ x.t()
    K2 = (1.0 - x) @ (1.0 - x.t())
    if reduction == 'sum':
        model.K += K.cpu().numpy() + K2.cpu().numpy()
    elif reduction == None:
        model.K.append(logdet(K.cpu().numpy() + K2.cpu().numpy()))
