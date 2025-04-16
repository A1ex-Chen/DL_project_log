def to_triton_onnx(self, dataloader, model):
    """ export the model to onnx and test correctness on dataloader """
    import onnx as local_onnx
    global onnx
    onnx = local_onnx
    import onnxruntime as local_onnxruntime
    global onnxruntime
    onnxruntime = local_onnxruntime
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
    final_model_path = os.path.join(version_folder, 'model.onnx')
    dynamic_axes = {}
    for input_name, input_shape in zip(input_names, input_shapes):
        dynamic_axes[input_name] = [i for i, x in enumerate(input_shape) if
            x == -1]
    for output_name, output_shape in zip(output_names, output_shapes):
        dynamic_axes[output_name] = [i for i, x in enumerate(output_shape) if
            x == -1]
    assert not model.training, 'internal error - model should be in eval() mode! '
    with torch.no_grad():
        torch.onnx.export(model, inputs[0], final_model_path, verbose=True,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=11)
    converted_model = onnx.load(final_model_path)
    onnx.checker.check_model(converted_model)
    self.name_onnx_nodes(final_model_path)
    converted_model = onnx.load(final_model_path)
    session = onnxruntime.InferenceSession(final_model_path, None)


    class ONNX_model:

        def __init__(self, session, input_names, device):
            self.session = session
            self.input_names = input_names

        def to_numpy(self, tensor):
            return tensor.detach().cpu().numpy(
                ) if tensor.requires_grad else tensor.cpu().numpy()

        def __call__(self, *inputs):
            inp = [(input_name, inputs[i]) for i, input_name in enumerate(
                self.input_names)]
            inp = {input_name: self.to_numpy(x) for input_name, x in inp}
            outputs = self.session.run(None, inp)
            outputs = [torch.from_numpy(output) for output in outputs]
            outputs = [output.to(device) for output in outputs]
            if len(outputs) == 1:
                outputs = outputs[0]
            return outputs
    model_onnx = ONNX_model(session, input_names, device)
    assert not model.training, 'internal error - model should be in eval() mode! '
    models = model, model_onnx
    outputs, time_model, outputs_onnx, time_model_onnx = self.lib.run_models(
        models, inputs)
    Error_stats = self.lib.compute_errors(outputs, outputs_onnx)
    self.lib.print_errors(Error_stats)
    print('time of error check of native model: ', time_model, 'seconds')
    print('time of error check of onnx model: ', time_model_onnx, 'seconds')
    print()
    config_filename = os.path.join(model_folder, 'config.pbtxt')
    self.lib.write_config(config_filename, input_shapes, input_types,
        output_shapes, output_types)
