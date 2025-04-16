def write_config(self, config_filename, input_shapes, input_types,
    output_shapes, output_types):
    """ writes TRTIS config file 
            :: config_filename :: the file to write the config file into
            :: input_shapes :: tuple of dynamic shapes of the input tensors
            :: input_types :: tuple of torch types of the input tensors
            :: output_shapes :: tuple of dynamic shapes of the output tensors
            :: output_types :: tuple of torch types of the output tensors
        """
    assert self.platform is not None, 'error - platform is not set'
    config_template = CONFIG_TEMPLATE
    input_template = INPUT_TEMPLATE
    optimization_template = MODEL_OPTIMIZATION_TEMPLATE
    accelerator_template = EXECUTION_ACCELERATOR_TEMPLATE
    spec_inputs = ''
    for i, (shape, typ) in enumerate(zip(input_shapes, input_types)):
        d = {'num': str(i), 'type': torch_type_to_triton_type[typ], 'dims':
            str([1]) if len(shape) == 1 else str(list(shape)[1:])}
        d['reshape'] = 'reshape: { shape: [ ] }' if len(shape) == 1 else ''
        spec_inputs += input_template.format_map(d)
    spec_inputs = spec_inputs[:-1]
    output_template = OUTPUT_TEMPLATE
    spec_outputs = ''
    for i, (shape, typ) in enumerate(zip(output_shapes, output_types)):
        d = {'num': str(i), 'type': torch_type_to_triton_type[typ], 'dims':
            str([1]) if len(shape) == 1 else str(list(shape)[1:])}
        d['reshape'] = 'reshape: { shape: [ ] }' if len(shape) == 1 else ''
        spec_outputs += output_template.format_map(d)
    spec_outputs = spec_outputs[:-1]
    batching_str = ''
    max_batch_size = self.args.triton_max_batch_size
    if self.args.triton_dyn_batching_delay >= 0:
        pref_batch_size = [int(max_batch_size / 2.0), max_batch_size]
        if self.args.triton_dyn_batching_delay > 0:
            dyn_batch_delay_str = (
                f'max_queue_delay_microseconds: {int(self.args.triton_dyn_batching_delay * 1000.0)}'
                )
        else:
            dyn_batch_delay_str = ''
        batching_str = (
            '\ndynamic_batching {{\n    preferred_batch_size: [{0}]\n    {1}\n}}'
            .format(', '.join([str(x) for x in pref_batch_size]),
            dyn_batch_delay_str))
    accelerator_str = ''
    d = {'execution_accelerator': accelerator_str, 'capture_cuda_graph':
        str(self.args.capture_cuda_graph)}
    optimization_str = optimization_template.format_map(d)
    config_values = {'model_name': self.args.triton_model_name, 'platform':
        self.platform, 'max_batch_size': max_batch_size, 'spec_inputs':
        spec_inputs, 'spec_outputs': spec_outputs, 'dynamic_batching':
        batching_str, 'model_optimizations': optimization_str, 'gpu_list':
        ', '.join([str(x) for x in range(torch.cuda.device_count())]),
        'engine_count': self.args.triton_engine_count}
    with open(config_filename, 'w') as file:
        final_config_str = config_template.format_map(config_values)
        final_config_str = remove_empty_lines(final_config_str)
        file.write(final_config_str)
