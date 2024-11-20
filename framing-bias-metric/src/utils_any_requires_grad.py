def any_requires_grad(model: nn.Module) ->bool:
    return any(grad_status(model))
