def set_grad(params, params_with_grad, scale=1.0):
    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(param.data.new().resize_(*param
                .data.size()))
        grad = param_w_grad.grad.data
        if scale is not None:
            grad /= scale
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            return True
        param.grad.data.copy_(grad)
    return False
