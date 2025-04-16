def tune_channel(ch):
    hist = cv2.calcHist([ch], [0], None, [n_bins], [0, n_bins])
    non_zero_hist = hist[hist != 0].reshape(-1)
    step = np.sum(non_zero_hist[:-1]) // (n_bins - 1)
    if step == 0:
        return ch
    n = np.empty_like(hist)
    n[0] = step // 2
    n[1:] = hist[:-1]
    table = (np.cumsum(n) // step).clip(0, 255).astype(np.uint8)
    return table[ch]
