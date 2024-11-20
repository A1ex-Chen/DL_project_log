def __init__(self, is_causal: bool, sliding_window: Optional[int]=None):
    self.is_causal = is_causal
    self.sliding_window = sliding_window
    if self.sliding_window is not None and self.sliding_window <= 0:
        raise ValueError(
            f'Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`'
            )
