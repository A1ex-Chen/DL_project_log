def get_flops(model, imgsz=640):
    """Return a YOLO model's FLOPs."""
    if not thop:
        return 0.0
    try:
        model = de_parallel(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]
        try:
            stride = max(int(model.stride.max()), 32) if hasattr(model,
                'stride') else 32
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0
                ] / 1000000000.0 * 2
            return flops * imgsz[0] / stride * imgsz[1] / stride
        except Exception:
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0
                ] / 1000000000.0 * 2
    except Exception:
        return 0.0
