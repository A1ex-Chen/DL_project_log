def _pad_1x1_to_3x3_tensor(self, kernel1x1):
    """Pads a 1x1 tensor to a 3x3 tensor."""
    if kernel1x1 is None:
        return 0
    else:
        return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
