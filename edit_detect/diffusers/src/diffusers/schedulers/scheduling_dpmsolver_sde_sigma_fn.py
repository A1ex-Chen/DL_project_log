def sigma_fn(_t: torch.Tensor) ->torch.Tensor:
    return _t.neg().exp()
