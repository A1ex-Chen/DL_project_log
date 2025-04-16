def _log_tensorboard_graph(trainer):
    """Log model graph to TensorBoard."""
    imgsz = trainer.args.imgsz
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
    p = next(trainer.model.parameters())
    im = torch.zeros((1, 3, *imgsz), device=p.device, dtype=p.dtype)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        warnings.simplefilter('ignore', category=torch.jit.TracerWarning)
        with contextlib.suppress(Exception):
            trainer.model.eval()
            WRITER.add_graph(torch.jit.trace(de_parallel(trainer.model), im,
                strict=False), [])
            LOGGER.info(f'{PREFIX}model graph visualization added ✅')
            return
        try:
            model = deepcopy(de_parallel(trainer.model))
            model.eval()
            model = model.fuse(verbose=False)
            for m in model.modules():
                if hasattr(m, 'export'):
                    m.export = True
                    m.format = 'torchscript'
            model(im)
            WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
            LOGGER.info(f'{PREFIX}model graph visualization added ✅')
        except Exception as e:
            LOGGER.warning(
                f'{PREFIX}WARNING ⚠️ TensorBoard graph visualization failure {e}'
                )
