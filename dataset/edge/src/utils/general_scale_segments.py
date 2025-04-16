def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None,
    normalize=False):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
            )
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - 
            img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    segments[:, 0] -= pad[0]
    segments[:, 1] -= pad[1]
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]
        segments[:, 1] /= img0_shape[0]
    return segments
