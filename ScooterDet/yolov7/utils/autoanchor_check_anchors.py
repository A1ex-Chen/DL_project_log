def check_anchors(dataset, model, thr=4.0, imgsz=640):
    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))
    wh = torch.tensor(np.concatenate([(l[:, 3:5] * s) for s, l in zip(
        shapes * scale, dataset.labels)])).float()

    def metric(k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1.0 / r).min(2)[0]
        best = x.max(1)[0]
        aat = (x > 1.0 / thr).float().sum(1).mean()
        bpr = (best > 1.0 / thr).float().mean()
        return bpr, aat
    anchors = m.anchor_grid.clone().cpu().view(-1, 2)
    bpr, aat = metric(anchors)
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}'
        , end='')
    if bpr < 0.98:
        print('. Attempting to improve anchors, please wait...')
        na = m.anchor_grid.numel() // 2
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr,
                gen=1000, verbose=False)
        except Exception as e:
            print(f'{prefix}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m
                .anchors)
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)
            check_anchor_order(m)
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m
                .anchors.device).view(-1, 1, 1)
            print(
                f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.'
                )
        else:
            print(
                f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.'
                )
    print('')
