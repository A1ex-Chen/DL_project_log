def notebook_init(verbose=True):
    print('Checking setup...')
    import os
    import shutil
    from utils.general import check_requirements, emojis, is_colab
    from utils.torch_utils import select_device
    check_requirements(('psutil', 'IPython'))
    import psutil
    from IPython import display
    if is_colab():
        shutil.rmtree('/content/sample_data', ignore_errors=True)
    if verbose:
        gb = 1 << 30
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage('/')
        display.clear_output()
        s = (
            f'({os.cpu_count()} CPUs, {ram / gb:.1f} GB RAM, {(total - free) / gb:.1f}/{total / gb:.1f} GB disk)'
            )
    else:
        s = ''
    select_device(newline=False)
    print(emojis(f'Setup complete ✅ {s}'))
    return display
