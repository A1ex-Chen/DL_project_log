def check_yolo(verbose=True, device=''):
    """Return a human-readable YOLO software and hardware summary."""
    import psutil
    from ultralytics.utils.torch_utils import select_device
    if IS_JUPYTER:
        if check_requirements('wandb', install=False):
            os.system('pip uninstall -y wandb')
        if IS_COLAB:
            shutil.rmtree('sample_data', ignore_errors=True)
    if verbose:
        gib = 1 << 30
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage('/')
        s = (
            f'({os.cpu_count()} CPUs, {ram / gib:.1f} GB RAM, {(total - free) / gib:.1f}/{total / gib:.1f} GB disk)'
            )
        with contextlib.suppress(Exception):
            from IPython import display
            display.clear_output()
    else:
        s = ''
    select_device(device=device, newline=False)
    LOGGER.info(f'Setup complete ✅ {s}')
