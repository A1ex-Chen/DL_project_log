def find_cuda():
    """Finds the CUDA install path."""
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        if sys.platform == 'win32':
            cuda_homes = glob.glob(
                'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
            if len(cuda_homes) == 0:
                cuda_home = ''
            else:
                cuda_home = cuda_homes[0]
        else:
            cuda_home = '/usr/local/cuda'
        if not os.path.exists(cuda_home):
            try:
                which = 'where' if sys.platform == 'win32' else 'which'
                nvcc = subprocess.check_output([which, 'nvcc']).decode(
                    ).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
            except Exception:
                cuda_home = None
    if cuda_home is None:
        raise RuntimeError("No CUDA runtime is found, using CUDA_HOME='{}'"
            .format(cuda_home))
    return cuda_home
