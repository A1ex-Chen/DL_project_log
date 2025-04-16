def _to_3d(self, hidden_states: torch.Tensor, height: int, weight: int
    ) ->torch.Tensor:
    return hidden_states.permute(0, 2, 3, 1).reshape(hidden_states.shape[0],
        height * weight, -1)
