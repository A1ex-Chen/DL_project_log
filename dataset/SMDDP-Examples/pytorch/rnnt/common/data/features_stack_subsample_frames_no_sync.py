def stack_subsample_frames_no_sync(x, x_lens, stacking=1, subsampling=1):
    """ Stacks frames together across feature dim, and then subsamples

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim * stacking, num_frames / subsampling

    """
    assert stacking == subsampling
    x = x.transpose(1, 2)
    T = x.size(1)
    padded = torch.nn.functional.pad(x, (0, 0, 0, (stacking - T % stacking) %
        stacking))
    B, T, H = padded.size()
    x = padded.reshape(B, T // stacking, -1)
    x = x.transpose(1, 2)
    x_lens = (x_lens.int() + stacking - 1) // stacking
    return x, x_lens
