def cuda_device_count() ->int:
    """
    Get the number of NVIDIA GPUs available in the environment.

    Returns:
        (int): The number of NVIDIA GPUs available.
    """
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=count',
            '--format=csv,noheader,nounits'], encoding='utf-8')
        first_line = output.strip().split('\n')[0]
        return int(first_line)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return 0
