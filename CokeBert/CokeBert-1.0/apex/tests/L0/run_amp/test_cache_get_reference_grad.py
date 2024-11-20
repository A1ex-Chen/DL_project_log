def get_reference_grad(i, w, ops):
    fp32_i = i.detach().clone().float()
    fp32_w = w.detach().clone().float().requires_grad_()
    loss = ops(fp32_i, fp32_w)
    loss.backward()
    return fp32_w.grad
