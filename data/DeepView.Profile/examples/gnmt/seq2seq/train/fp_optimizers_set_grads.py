@staticmethod
def set_grads(params, params_with_grad):
    """
        Copies gradients from param_with_grad to params

        :param params: dst parameters
        :param params_with_grad: src parameters
        """
    for param, param_w_grad in zip(params, params_with_grad):
        if param.grad is None:
            param.grad = torch.nn.Parameter(torch.empty_like(param))
        param.grad.data.copy_(param_w_grad.grad.data)
