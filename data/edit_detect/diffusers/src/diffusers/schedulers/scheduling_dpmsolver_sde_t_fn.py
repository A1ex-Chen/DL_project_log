def t_fn(_sigma: torch.Tensor) ->torch.Tensor:
    return _sigma.log().neg()
