# YOLOv3 🚀 by Ultralytics, AGPL-3.0 license
"""AutoAnchor utils."""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils import TryExcept
from utils.general import LOGGER, TQDM_BAR_FORMAT, colorstr

PREFIX = colorstr("AutoAnchor: ")




@TryExcept(f"{PREFIX}ERROR")



    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    anchors = m.anchors.clone() * stride  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f"\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). "
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f"{s}Current anchors are a good fit to dataset ✅")
    else:
        LOGGER.info(f"{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...")
        na = m.anchors.numel() // 2  # number of anchors
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f"{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)"
        else:
            s = f"{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)"
        LOGGER.info(s)


def kmean_anchors(dataset="./data/coco128.yaml", n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """
    Creates kmeans-evolved anchors from training dataset.

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




    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors="ignore") as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels

        dataset = LoadImagesAndLabels(data_dict["train"], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f"{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size")
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        LOGGER.info(f"{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...")
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f"{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init")
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format=TQDM_BAR_FORMAT)  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f"{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}"
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)