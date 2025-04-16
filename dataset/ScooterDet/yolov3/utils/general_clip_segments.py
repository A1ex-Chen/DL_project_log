def clip_segments(segments, shape):
    if isinstance(segments, torch.Tensor):
        segments[:, 0].clamp_(0, shape[1])
        segments[:, 1].clamp_(0, shape[0])
    else:
        segments[:, 0] = segments[:, 0].clip(0, shape[1])
        segments[:, 1] = segments[:, 1].clip(0, shape[0])
