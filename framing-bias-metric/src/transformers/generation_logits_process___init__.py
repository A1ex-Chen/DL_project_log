def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor],
    List[int]], num_beams: int):
    self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
    self._num_beams = num_beams
