def kmean_anchors(path='./data/coco.yaml', n=9, img_size=640, thr=4.0, gen=
    1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    thr = 1.0 / thr
    prefix = colorstr('autoanchor: ')

    def metric(k, wh):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1.0 / r).min(2)[0]
        return x, x.max(1)[0]

    def anchor_fitness(k):
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()

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
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < 
                len(k) - 1 else '\n')
        return k
    if isinstance(path, str):
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True,
            rect=True)
    else:
        dataset = path
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([(l[:, 3:5] * s) for s, l in zip(shapes, dataset.
        labels)])
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(
            f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.'
            )
    wh = wh0[(wh0 >= 2.0).any(1)]
    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)
    k, dist = kmeans(wh / s, n, iter=30)
    assert len(k) == n, print(
        f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}'
        )
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)
    wh0 = torch.tensor(wh0, dtype=torch.float32)
    k = print_results(k)
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1
    pbar = tqdm(range(gen), desc=
        f'{prefix}Evolving anchors with Genetic Algorithm:')
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1
                ).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = (
                f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
                )
            if verbose:
                print_results(k)
    return print_results(k)
