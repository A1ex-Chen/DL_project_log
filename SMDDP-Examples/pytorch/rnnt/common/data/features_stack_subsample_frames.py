def stack_subsample_frames(x, x_lens, stacking=1, subsampling=1):
    """ Stacks frames together across feature dim, and then subsamples

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim * stacking, num_frames / subsampling

    """
    seq = [x]
    for n in range(1, stacking):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    x = torch.cat(seq, dim=1)[:, :, ::subsampling]
    if subsampling > 1:
        x_lens = torch.ceil(x_lens.float() / subsampling).int()
        if x.size(2) > x_lens.max().item():
            assert abs(x.size(2) - x_lens.max().item()) <= 1
            x = x[:, :, :x_lens.max().item()]
    return x, x_lens
