def broadcast_val(elements, shapes):
    ret_grads = []
    for e, sh in zip(elements, shapes):
        ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e],
            dim=0).to(device))
    return ret_grads
