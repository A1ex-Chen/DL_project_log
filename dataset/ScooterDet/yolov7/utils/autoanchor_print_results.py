def print_results(k):
    k = k[np.argsort(k.prod(1))]
    x, best = metric(k, wh0)
    bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n
    print(
        f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr'
        )
    print(
        f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, past_thr={x[x > thr].mean():.3f}-mean: '
        , end='')
    for i, x in enumerate(k):
        print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) -
            1 else '\n')
    return k
