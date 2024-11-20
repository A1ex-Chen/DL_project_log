def _device_updater(op, *args, **kwargs):
    return {'CopyCPUToGPU': _copy_cpu_to_gpu_updater, 'CopyGPUToCPU':
        _copy_gpu_to_cpu_updater}.get(op.type, _other_ops_updater)(op, *
        args, **kwargs)
