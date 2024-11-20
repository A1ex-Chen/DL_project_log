def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ['--threads', '4']
