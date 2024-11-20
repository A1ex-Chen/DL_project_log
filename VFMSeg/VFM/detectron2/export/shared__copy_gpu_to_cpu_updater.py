def _copy_gpu_to_cpu_updater(op, input_types, output_types):
    if input_types[0] == _CPU_STR or output_types[0] == _GPU_STR:
        _updater_raise(op, input_types, output_types)
    return [_GPU_STR], [_CPU_STR]
