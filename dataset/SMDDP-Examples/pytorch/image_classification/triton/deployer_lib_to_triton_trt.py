def to_triton_trt(self, dataloader, model):
    """ export the model to trt and test correctness on dataloader """
    import tensorrt as trt
    if self.args.triton_no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    model.to(device)
    model.eval()
    assert not model.training, 'internal error - model should be in eval() mode! '
    inputs = self.lib.prepare_inputs(dataloader, device)
    outputs = []
    for input in inputs:
        with torch.no_grad():
            output = model(*input)
        if type(output) is torch.Tensor:
            output = [output]
        outputs.append(output)
    input_shapes = self.lib.get_tuple_of_dynamic_shapes(inputs)
    output_shapes = self.lib.get_tuple_of_dynamic_shapes(outputs)
    input_types = [x.dtype for x in inputs[0]]
    output_types = [x.dtype for x in outputs[0]]
    rng = range(len(input_types))
    input_names = [('input__' + str(num)) for num in rng]
    rng = range(len(output_types))
    output_names = [('output__' + str(num)) for num in rng]
    model_folder = os.path.join(self.args.save_dir, self.args.triton_model_name
        )
    version_folder = os.path.join(model_folder, str(self.args.
        triton_model_version))
    if not os.path.exists(version_folder):
        os.makedirs(version_folder)
    final_model_path = os.path.join(version_folder, 'model.plan')
    dynamic_axes = {}
    for input_name, shape in zip(input_names, input_shapes):
        dynamic_axes[input_name] = [i for i, x in enumerate(shape) if x == -1]
    for output_name, shape in zip(output_names, output_shapes):
        dynamic_axes[output_name] = [i for i, x in enumerate(shape) if x == -1]
    with torch.no_grad():
        torch.onnx.export(model, inputs[0], final_model_path, verbose=False,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=11)
    min_shapes = self.lib.get_tuple_of_min_shapes(inputs)
    opt_shapes = self.lib.get_tuple_of_opt_shapes(inputs)
    max_shapes = self.lib.get_tuple_of_max_shapes(inputs)
    zipped = zip(input_names, min_shapes, opt_shapes, max_shapes)
    shapes = []
    for name, min_shape, opt_shape, max_shape in zipped:
        d = {'name': name, 'min': min_shape, 'opt': opt_shape, 'max': max_shape
            }
        shapes.append(d)
    engine = self.lib.build_trt_engine(final_model_path, shapes)
    assert engine is not None, ' trt export failure '
    with open(final_model_path, 'wb') as f:
        f.write(engine.serialize())
    engine = self.lib.load_engine(final_model_path)


    class TRT_model:

        def __init__(self, engine, input_names, output_names, output_types,
            device):
            self.engine = engine
            self.context = self.engine.create_execution_context()
            self.input_names = input_names
            self.output_names = output_names
            self.output_types = output_types
            self.device = device

        def is_dimension_dynamic(self, dim):
            return dim is None or dim <= 0

        def is_shape_dynamic(self, shape):
            return any([self.is_dimension_dynamic(dim) for dim in shape])

        def __call__(self, *inputs):
            input_shapes = [x.shape for x in inputs]
            bindings = [None] * self.engine.num_bindings
            zipped = zip(self.input_names, inputs)
            for key, input in zipped:
                idx = self.engine.get_binding_index(key)
                bindings[idx] = input.data_ptr()
                if self.engine.is_shape_binding(idx) and self.is_shape_dynamic(
                    self.context.get_shape(idx)):
                    self.context.set_shape_input(idx, input)
                elif self.is_shape_dynamic(self.engine.get_binding_shape(idx)):
                    self.context.set_binding_shape(idx, input.shape)
            assert self.context.all_binding_shapes_specified, 'trt error'
            assert self.context.all_shape_inputs_specified, 'trt error'
            outputs = []
            zipped = zip(self.output_names, self.output_types)
            for key, dtype in zipped:
                idx = self.engine.get_binding_index(key)
                shape = self.context.get_binding_shape(idx)
                shape = tuple(shape)
                assert -1 not in shape, 'trt error'
                tensor = torch.zeros(shape, dtype=dtype, device=self.device)
                outputs.append(tensor)
                bindings[idx] = outputs[-1].data_ptr()
            self.context.execute_v2(bindings=bindings)
            if len(outputs) == 1:
                outputs = outputs[0]
            return outputs
    model_trt = TRT_model(engine, input_names, output_names, output_types,
        device)
    assert not model.training, 'internal error - model should be in eval() mode! '
    models = model, model_trt
    outputs, time_model, outputs_trt, time_model_trt = self.lib.run_models(
        models, inputs)
    Error_stats = self.lib.compute_errors(outputs, outputs_trt)
    self.lib.print_errors(Error_stats)
    print('time of error check of native model: ', time_model, 'seconds')
    print('time of error check of trt model: ', time_model_trt, 'seconds')
    print()
    config_filename = os.path.join(model_folder, 'config.pbtxt')
    self.lib.write_config(config_filename, input_shapes, input_types,
        output_shapes, output_types)
