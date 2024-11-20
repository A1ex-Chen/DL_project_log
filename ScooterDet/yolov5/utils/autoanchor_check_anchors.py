@TryExcept(f'{PREFIX}ERROR')
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = torch.tensor(np.concatenate([(l[:, 3:5] * s) for s, l in zip(
        shapes * scale, dataset.labels)])).float()

    def metric(k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]
        best = x.max(1)[0]
        aat = (x > 1 / thr).float().sum(1).mean()
        bpr = (best > 1 / thr).float().mean()
        return bpr, aat
    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)
    anchors = m.anchors.clone() * stride
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = (
        f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
        )
    if bpr > 0.98:
        LOGGER.info(f'{s}Current anchors are a good fit to dataset ✅')
    else:
        LOGGER.info(
            f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...'
            )
        na = m.anchors.numel() // 2
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen
            =1000, verbose=False)
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m
                .anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)
            m.anchors /= stride
            s = (
                f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
                )
        else:
            s = (
                f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
                )
        LOGGER.info(s)
