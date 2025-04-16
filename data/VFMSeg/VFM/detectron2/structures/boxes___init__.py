def __init__(self, tensor: torch.Tensor):
    """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
    device = tensor.device if isinstance(tensor, torch.Tensor
        ) else torch.device('cpu')
    tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32, device=device)
    assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()
    self.tensor = tensor
