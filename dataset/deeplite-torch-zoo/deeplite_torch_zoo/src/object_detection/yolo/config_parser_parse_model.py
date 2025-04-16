def parse_model(d, ch, activation_type, depth_mul=None, width_mul=None,
    max_channels=None, yolo_channel_divisor=8):
    LOGGER.info(
        f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}"
        )
    anchors, nc = d['anchors'], d['nc']
    gd = depth_mul
    gw = width_mul
    activation_type = activation_type if activation_type is not None else d[
        'activation_type']
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        try:
            m = eval(m) if isinstance(m, str) else m
        except:
            m = eval(f'YOLO{m}') if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass
        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in VARIABLE_CHANNEL_BLOCKS.registry_dict.values():
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(min(c2, max_channels) * gw,
                    yolo_channel_divisor)
            args = [c1, c2, *args[1:]]
            if m in EXPANDABLE_BLOCKS.registry_dict.values():
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is ADD or m is Shortcut:
            c2 = ch[f[0]]
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is DetectX:
            args.append([ch[x] for x in f])
        elif m is DetectV8:
            args = args[:1]
            args.append([ch[x] for x in f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is ReOrg or m is DWT:
            c2 = ch[f] * 4
        else:
            c2 = ch[f]
        kwargs = dict()
        if 'act' in inspect.signature(m).parameters:
            kwargs.update({'act': activation_type})
        m_ = nn.Sequential(*(m(*args, **kwargs) for _ in range(n))
            ) if n > 1 else m(*args, **kwargs)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        LOGGER.info(
            f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x !=
            -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
