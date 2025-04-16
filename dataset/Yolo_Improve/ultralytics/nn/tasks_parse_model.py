def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple',
        'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(
                f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]
    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")
    if verbose:
        LOGGER.info(
            f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}"
            )
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        try:
            if m == 'node_mode':
                m = d[m]
                if len(args) > 0:
                    if args[0] == 'head_channel':
                        args[0] = int(d[args[0]])
            t = m
            m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]
        except:
            pass
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals(
                        ) else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n
        if m in {Classify, Conv, ConvTranspose, GhostConv, Bottleneck,
            GhostBottleneck, SPP, SPPF, DWConv, Focus, BottleneckCSP, C1,
            C2, C2f, RepNCSPELAN4, ELAN1, ADown, AConv, SPPELAN, C2fAttn,
            C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x,
            RepC3, PSA, SCDown, C2fCIB}:
            if args[0] == 'head_channel':
                args[0] = d[args[0]]
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) *
                    width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 
                    32)) * width, 1) if args[2] > 1 else args[2])
            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost,
                C3x, RepC3, C2fCIB}:
                args.insert(2, n)
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB,
            ImagePoolingAttn, v10Detect}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m is Fusion:
            args[0] = d[args[0]]
            c1 = [ch[x] for x in f]
            c2 = sum(ch[x] for x in f) if args[0] == 'concat' else ch[f[0]]
            args = [c1, args[0]]
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args
            )
        t = str(m)[8:-2].replace('__main__.', '')
        m.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        if verbose:
            LOGGER.info(
                f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}'
                )
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x !=
            -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
