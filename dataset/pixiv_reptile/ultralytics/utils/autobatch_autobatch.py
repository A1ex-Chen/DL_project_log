def autobatch(model, imgsz=640, fraction=0.6, batch_size=DEFAULT_CFG.batch):
    """
    Automatically estimate the best YOLO batch size to use a fraction of the available CUDA memory.

    Args:
        model (torch.nn.module): YOLO model to compute batch size for.
        imgsz (int, optional): The image size used as input for the YOLO model. Defaults to 640.
        fraction (float, optional): The fraction of available CUDA memory to use. Defaults to 0.60.
        batch_size (int, optional): The default batch size to use if an error is detected. Defaults to 16.

    Returns:
        (int): The optimal batch size.
    """
    prefix = colorstr('AutoBatch: ')
    LOGGER.info(
        f'{prefix}Computing optimal batch size for imgsz={imgsz} at {fraction * 100}% CUDA memory utilization.'
        )
    device = next(model.parameters()).device
    if device.type in {'cpu', 'mps'}:
        LOGGER.info(
            f'{prefix} ⚠️ intended for CUDA devices, using default batch-size {batch_size}'
            )
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(
            f'{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}'
            )
        return batch_size
    gb = 1 << 30
    d = str(device).upper()
    properties = torch.cuda.get_device_properties(device)
    t = properties.total_memory / gb
    r = torch.cuda.memory_reserved(device) / gb
    a = torch.cuda.memory_allocated(device) / gb
    f = t - (r + a)
    LOGGER.info(
        f'{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free'
        )
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
        y = [x[2] for x in results if x]
        p = np.polyfit(batch_sizes[:len(y)], y, deg=1)
        b = int((f * fraction - p[1]) / p[0])
        if None in results:
            i = results.index(None)
            if b >= batch_sizes[i]:
                b = batch_sizes[max(i - 1, 0)]
        if b < 1 or b > 1024:
            b = batch_size
            LOGGER.info(
                f'{prefix}WARNING ⚠️ CUDA anomaly detected, using default batch-size {batch_size}.'
                )
        fraction = (np.polyval(p, b) + r + a) / t
        LOGGER.info(
            f'{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅'
            )
        return b
    except Exception as e:
        LOGGER.warning(
            f'{prefix}WARNING ⚠️ error detected: {e},  using default batch-size {batch_size}.'
            )
        return batch_size
