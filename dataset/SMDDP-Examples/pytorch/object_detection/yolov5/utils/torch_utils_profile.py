def profile(input, ops, n=10, device=None):
    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    print(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}{'input':>24s}{'output':>24s}"
        )
    for x in (input if isinstance(input, list) else [input]):
        x = x.to(device)
        x.requires_grad = True
        for m in (ops if isinstance(ops, list) else [ops]):
            m = m.to(device) if hasattr(m, 'to') else m
            m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor
                ) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]
            try:
                flops = thop.profile(m, inputs=(x,), verbose=False)[0
                    ] / 1000000000.0 * 2
            except Exception:
                flops = 0
            try:
                for _ in range(n):
                    t[0] = time_sync()
                    y = m(x)
                    t[1] = time_sync()
                    try:
                        _ = (sum(yi.sum() for yi in y) if isinstance(y,
                            list) else y).sum().backward()
                        t[2] = time_sync()
                    except Exception:
                        t[2] = float('nan')
                    tf += (t[1] - t[0]) * 1000 / n
                    tb += (t[2] - t[1]) * 1000 / n
                mem = torch.cuda.memory_reserved(
                    ) / 1000000000.0 if torch.cuda.is_available() else 0
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor
                    ) else 'list' for x in (x, y))
                p = sum(x.numel() for x in m.parameters()) if isinstance(m,
                    nn.Module) else 0
                print(
                    f'{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{str(s_in):>24s}{str(s_out):>24s}'
                    )
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                print(e)
                results.append(None)
            torch.cuda.empty_cache()
    return results
