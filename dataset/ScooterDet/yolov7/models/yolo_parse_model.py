def parse_model(d, ch):
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params',
        'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d[
        'width_multiple']
    na = len(anchors[0]) // 2 if isinstance(anchors, list) else anchors
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv,
            GhostConv, RepConv, RepConv_OREPA, DownC, SPP, SPPF, SPPCSPC,
            GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv,
            Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC,
            RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB,
            RepBottleneckCSPC, Res, ResCSPA, ResCSPB, ResCSPC, RepRes,
            RepResCSPA, RepResCSPB, RepResCSPC, ResX, ResXCSPA, ResXCSPB,
            ResXCSPC, RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC, Ghost,
            GhostCSPA, GhostCSPB, GhostCSPC, SwinTransformerBlock, STCSPA,
            STCSPB, STCSPC, SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC, BottleneckCSPA,
                BottleneckCSPB, BottleneckCSPC, RepBottleneckCSPA,
                RepBottleneckCSPB, RepBottleneckCSPC, ResCSPA, ResCSPB,
                ResCSPC, RepResCSPA, RepResCSPB, RepResCSPC, ResXCSPA,
                ResXCSPB, ResXCSPC, RepResXCSPA, RepResXCSPB, RepResXCSPC,
                GhostCSPA, GhostCSPB, GhostCSPC, STCSPA, STCSPB, STCSPC,
                ST2CSPA, ST2CSPB, ST2CSPC]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args
            )
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x !=
            -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)
