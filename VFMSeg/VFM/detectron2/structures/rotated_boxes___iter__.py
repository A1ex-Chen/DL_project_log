@torch.jit.unused
def __iter__(self):
    """
        Yield a box as a Tensor of shape (5,) at a time.
        """
    yield from self.tensor
