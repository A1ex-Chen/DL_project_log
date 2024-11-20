def print_results(k, verbose=True):
    k = k[np.argsort(k.prod(1))]
    x, best = metric(k, wh0)
    bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n
    s = f"""{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr
{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, past_thr={x[x > thr].mean():.3f}-mean: """
    for x in k:
        s += '%i,%i, ' % (round(x[0]), round(x[1]))
    if verbose:
        LOGGER.info(s[:-2])
    return k
