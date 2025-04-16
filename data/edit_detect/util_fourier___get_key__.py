def __get_key__(self, key: torch.Tensor) ->int:
    return hash(tuple(key.reshape(-1).tolist()))
