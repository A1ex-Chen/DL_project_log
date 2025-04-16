def get_flops_with_torch_profiler(model, imgsz=640):
    """Compute model FLOPs (thop package alternative, but 2-10x slower unfortunately)."""
    if not TORCH_2_0:
        return 0.0
    model = de_parallel(model)
    p = next(model.parameters())
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]
    try:
        stride = (max(int(model.stride.max()), 32) if hasattr(model,
            'stride') else 32) * 2
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1000000000.0
        flops = flops * imgsz[0] / stride * imgsz[1] / stride
    except Exception:
        im = torch.empty((1, p.shape[1], *imgsz), device=p.device)
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1000000000.0
    return flops
