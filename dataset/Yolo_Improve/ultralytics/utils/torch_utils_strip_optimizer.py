def strip_optimizer(f: Union[str, Path]='best.pt', s: str='') ->None:
    """
    Strip optimizer from 'f' to finalize training, optionally save as 's'.

    Args:
        f (str): file path to model to strip the optimizer from. Default is 'best.pt'.
        s (str): file path to save the model with stripped optimizer to. If not provided, 'f' will be overwritten.

    Returns:
        None

    Example:
        ```python
        from pathlib import Path
        from ultralytics.utils.torch_utils import strip_optimizer

        for f in Path('path/to/model/checkpoints').rglob('*.pt'):
            strip_optimizer(f)
        ```
    """
    try:
        x = torch.load(f, map_location=torch.device('cpu'))
        assert isinstance(x, dict), 'checkpoint is not a Python dictionary'
        assert 'model' in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(
            f'WARNING ⚠️ Skipping {f}, not a valid Ultralytics model: {e}')
        return
    updates = {'date': datetime.now().isoformat(), 'version': __version__,
        'license': 'AGPL-3.0 License (https://ultralytics.com/license)',
        'docs': 'https://docs.ultralytics.com'}
    if x.get('ema'):
        x['model'] = x['ema']
    if hasattr(x['model'], 'args'):
        x['model'].args = dict(x['model'].args)
    if hasattr(x['model'], 'criterion'):
        x['model'].criterion = None
    x['model'].half()
    for p in x['model'].parameters():
        p.requires_grad = False
    args = {**DEFAULT_CFG_DICT, **x.get('train_args', {})}
    for k in ('optimizer', 'best_fitness', 'ema', 'updates'):
        x[k] = None
    x['epoch'] = -1
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    torch.save({**updates, **x}, s or f, use_dill=False)
    mb = os.path.getsize(s or f) / 1000000.0
    LOGGER.info(
        f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB"
        )
