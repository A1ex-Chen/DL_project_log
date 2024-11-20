def _copy_cpu_to_gpu_updater(op, input_types, output_types):
    if input_types[0] == _GPU_STR or output_types[0] == _CPU_STR:
        _updater_raise(op, input_types, output_types)
    return [_CPU_STR], [_GPU_STR]
