def profile(x, ops, n=100, device=None):
    device = device or torch.device('cuda:0' if torch.cuda.is_available() else
        'cpu')
    x = x.to(device)
    x.requires_grad = True
    print(torch.__version__, device.type, torch.cuda.get_device_properties(
        0) if device.type == 'cuda' else '')
    print(
        f"""
{'Params':>12s}{'GFLOPS':>12s}{'forward (ms)':>16s}{'backward (ms)':>16s}{'input':>24s}{'output':>24s}"""
        )
    for m in (ops if isinstance(ops, list) else [ops]):
        m = m.to(device) if hasattr(m, 'to') else m
        m = m.half() if hasattr(m, 'half') and isinstance(x, torch.Tensor
            ) and x.dtype is torch.float16 else m
        dtf, dtb, t = 0.0, 0.0, [0.0, 0.0, 0.0]
        try:
            flops = thop.profile(m, inputs=(x,), verbose=False)[0
                ] / 1000000000.0 * 2
        except:
            flops = 0
        for _ in range(n):
            t[0] = time_synchronized()
            y = m(x)
            t[1] = time_synchronized()
            try:
                _ = y.sum().backward()
                t[2] = time_synchronized()
            except:
                t[2] = float('nan')
            dtf += (t[1] - t[0]) * 1000 / n
            dtb += (t[2] - t[1]) * 1000 / n
        s_in = tuple(x.shape) if isinstance(x, torch.Tensor) else 'list'
        s_out = tuple(y.shape) if isinstance(y, torch.Tensor) else 'list'
        p = sum(list(x.numel() for x in m.parameters())) if isinstance(m,
            nn.Module) else 0
        print(
            f'{p:12}{flops:12.4g}{dtf:16.4g}{dtb:16.4g}{str(s_in):>24s}{str(s_out):>24s}'
            )
