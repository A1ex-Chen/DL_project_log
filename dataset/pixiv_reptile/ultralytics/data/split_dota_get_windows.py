def get_windows(im_size, crop_sizes=(1024,), gaps=(200,), im_rate_thr=0.6,
    eps=0.01):
    """
    Get the coordinates of windows.

    Args:
        im_size (tuple): Original image size, (h, w).
        crop_sizes (List(int)): Crop size of windows.
        gaps (List(int)): Gap between crops.
        im_rate_thr (float): Threshold of windows areas divided by image ares.
        eps (float): Epsilon value for math operations.
    """
    h, w = im_size
    windows = []
    for crop_size, gap in zip(crop_sizes, gaps):
        assert crop_size > gap, f'invalid crop_size gap pair [{crop_size} {gap}]'
        step = crop_size - gap
        xn = 1 if w <= crop_size else ceil((w - crop_size) / step + 1)
        xs = [(step * i) for i in range(xn)]
        if len(xs) > 1 and xs[-1] + crop_size > w:
            xs[-1] = w - crop_size
        yn = 1 if h <= crop_size else ceil((h - crop_size) / step + 1)
        ys = [(step * i) for i in range(yn)]
        if len(ys) > 1 and ys[-1] + crop_size > h:
            ys[-1] = h - crop_size
        start = np.array(list(itertools.product(xs, ys)), dtype=np.int64)
        stop = start + crop_size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)
    im_in_wins = windows.copy()
    im_in_wins[:, 0::2] = np.clip(im_in_wins[:, 0::2], 0, w)
    im_in_wins[:, 1::2] = np.clip(im_in_wins[:, 1::2], 0, h)
    im_areas = (im_in_wins[:, 2] - im_in_wins[:, 0]) * (im_in_wins[:, 3] -
        im_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * (windows[:, 3] - windows[
        :, 1])
    im_rates = im_areas / win_areas
    if not (im_rates > im_rate_thr).any():
        max_rate = im_rates.max()
        im_rates[abs(im_rates - max_rate) < eps] = 1
    return windows[im_rates > im_rate_thr]
