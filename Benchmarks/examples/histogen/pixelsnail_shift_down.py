def shift_down(input, size=1):
    return F.pad(input, [0, 0, size, 0])[:, :, :input.shape[2], :]
