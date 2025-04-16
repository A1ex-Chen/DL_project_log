def parse_model(d, ch, model, imgsz):
    LOGGER.info(
        f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}"
        )
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d[
        'width_multiple']
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m_str = m
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass
        n = max(round(n * gd), 1) if n > 1 else n
        if m in [nn.Conv2d, Conv, DWConv, DWConvTranspose2d, Bottleneck,
            SPP, SPPF, MixConv2d, Focus, CrossConv, BottleneckCSP, C3, C3x]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3x]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[-1 if x == -1 else x + 1] for x in f)
        elif m in [Detect, Segment]:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
            args.append(imgsz)
        else:
            c2 = ch[f]
        tf_m = eval('TF' + m_str.replace('nn.', ''))
        m_ = keras.Sequential([tf_m(*args, w=model.model[i][j]) for j in
            range(n)]) if n > 1 else tf_m(*args, w=model.model[i])
        torch_m_ = nn.Sequential(*(m(*args) for _ in range(n))
            ) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum(x.numel() for x in torch_m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info(
            f'{i:>3}{str(f):>18}{str(n):>3}{np:>10}  {t:<40}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x !=
            -1)
        layers.append(m_)
        ch.append(c2)
    return keras.Sequential(layers), sorted(save)
