def notebook_init(verbose=True):
    print('Checking setup...')
    import os
    import shutil
    from utils.general import check_font, check_requirements, is_colab
    from utils.torch_utils import select_device
    check_font()
    import psutil
    if is_colab():
        shutil.rmtree('/content/sample_data', ignore_errors=True)
    display = None
    if verbose:
        gb = 1 << 30
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage('/')
        with contextlib.suppress(Exception):
            from IPython import display
            display.clear_output()
        s = (
            f'({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)'
            )
    else:
        s = ''
    select_device(newline=False)
    print(emojis(f'Setup complete ✅ {s}'))
    return display
