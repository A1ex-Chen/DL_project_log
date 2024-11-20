def shift_right(input, size=1):
    return F.pad(input, [size, 0, 0, 0])[:, :, :, :input.shape[3]]
