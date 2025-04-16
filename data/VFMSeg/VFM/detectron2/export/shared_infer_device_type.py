def infer_device_type(predict_net: caffe2_pb2.NetDef, known_status: Dict[
    Tuple[str, int], Any], device_name_style: str='caffe2') ->Dict[Tuple[
    str, int], str]:
    """Return the device type ("cpu" or "gpu"/"cuda") of each (versioned) blob"""
    assert device_name_style in ['caffe2', 'pytorch']
    _CPU_STR = 'cpu'
    _GPU_STR = 'gpu' if device_name_style == 'caffe2' else 'cuda'

    def _copy_cpu_to_gpu_updater(op, input_types, output_types):
        if input_types[0] == _GPU_STR or output_types[0] == _CPU_STR:
            _updater_raise(op, input_types, output_types)
        return [_CPU_STR], [_GPU_STR]

    def _copy_gpu_to_cpu_updater(op, input_types, output_types):
        if input_types[0] == _CPU_STR or output_types[0] == _GPU_STR:
            _updater_raise(op, input_types, output_types)
        return [_GPU_STR], [_CPU_STR]

    def _other_ops_updater(op, input_types, output_types):
        non_none_types = [x for x in input_types + output_types if x is not
            None]
        if len(non_none_types) > 0:
            the_type = non_none_types[0]
            if not all(x == the_type for x in non_none_types):
                _updater_raise(op, input_types, output_types)
        else:
            the_type = None
        return [the_type for _ in op.input], [the_type for _ in op.output]

    def _device_updater(op, *args, **kwargs):
        return {'CopyCPUToGPU': _copy_cpu_to_gpu_updater, 'CopyGPUToCPU':
            _copy_gpu_to_cpu_updater}.get(op.type, _other_ops_updater)(op,
            *args, **kwargs)
    return _generic_status_identifier(predict_net, _device_updater,
        known_status)
