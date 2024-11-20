def epsilon_distance(epsilon: torch.Tensor, pred_epsilon: torch.Tensor
    ) ->torch.Tensor:
    return (epsilon ** 2 - (epsilon - pred_epsilon) ** 2).mean(list(range(
        epsilon.dim()))[1:])
