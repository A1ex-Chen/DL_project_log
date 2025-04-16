def copy_tensor(src, dst):
    assert dst.numel() == src.numel()
    if move_eos_to_beginning:
        assert src[-1] == eos_idx
        dst[0] = eos_idx
        dst[1:] = src[:-1]
    else:
        dst.copy_(src)
