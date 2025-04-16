@staticmethod
def mse(x0: torch.Tensor, pred_x0: torch.Tensor, reduction: str
    ) ->torch.Tensor:
    return ((x0 - pred_x0) ** 2).sqrt().mean(list(range(x0.dim()))[1:])
