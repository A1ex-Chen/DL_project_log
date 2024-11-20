def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0,
    gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
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
    from scipy.cluster.vq import kmeans
    npr = np.random
    thr = 1 / thr

    def metric(k, wh):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]
        return x, x.max(1)[0]

    def anchor_fitness(k):
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()

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
    if isinstance(dataset, str):
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
        from utils.dataloaders import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True,
            rect=True)
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([(l[:, 3:5] * s) for s, l in zip(shapes, dataset.
        labels)])
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(
            f'{PREFIX}WARNING: Extremely small objects found: {i} of {len(wh0)} labels are < 3 pixels in size'
            )
    wh = wh0[(wh0 >= 2.0).any(1)]
    try:
        LOGGER.info(
            f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)
        s = wh.std(0)
        k = kmeans(wh / s, n, iter=30)[0] * s
        assert n == len(k)
    except Exception:
        LOGGER.warning(
            f'{PREFIX}WARNING: switching strategies from kmeans to random init'
            )
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1
    pbar = tqdm(range(gen), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) *
                s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = (
                f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
                )
            if verbose:
                print_results(k, verbose)
    return print_results(k)
