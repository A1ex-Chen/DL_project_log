def log_tensorboard_graph(tb, model, imgsz=(640, 640)):
    try:
        p = next(model.parameters())
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        im = torch.zeros((1, 3, *imgsz)).to(p.device).type_as(p)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tb.add_graph(torch.jit.trace(de_parallel(model), im, strict=
                False), [])
    except Exception as e:
        LOGGER.warning(
            f'WARNING ⚠️ TensorBoard graph visualization failure {e}')
