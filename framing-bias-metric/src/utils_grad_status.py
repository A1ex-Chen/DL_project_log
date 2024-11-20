def grad_status(model: nn.Module) ->Iterable:
    return (par.requires_grad for par in model.parameters())
