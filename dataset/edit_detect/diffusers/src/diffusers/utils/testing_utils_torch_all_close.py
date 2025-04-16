def torch_all_close(a, b, *args, **kwargs):
    if not is_torch_available():
        raise ValueError('PyTorch needs to be installed to use this function.')
    if not torch.allclose(a, b, *args, **kwargs):
        assert False, f'Max diff is absolute {(a - b).abs().max()}. Diff tensor is {(a - b).abs()}.'
    return True
