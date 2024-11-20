def to_triton_torchscript(self, dataloader, model):
    """ export the model to torchscript and test correctness on dataloader """
    if self.args.triton_no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    model.to(device)
    model.eval()
    assert not model.training, 'internal error - model should be in eval() mode! '
    inputs = self.lib.prepare_inputs(dataloader, device)
    input_shapes = self.lib.get_tuple_of_dynamic_shapes(inputs)
    input_types = [x.dtype for x in inputs[0]]
    model_folder = os.path.join(self.args.save_dir, self.args.triton_model_name
        )
    version_folder = os.path.join(model_folder, str(self.args.
        triton_model_version))
    if not os.path.exists(version_folder):
        os.makedirs(version_folder)
    final_model_path = os.path.join(version_folder, 'model.pt')
    with torch.no_grad():
        if self.args.ts_trace:
            model_ts = torch.jit.trace(model, inputs[0])
        if self.args.ts_script:
            model_ts = torch.jit.script(model)
    torch.jit.save(model_ts, final_model_path)
    model_ts = torch.jit.load(final_model_path)
    model_ts.eval()
    assert not model.training, 'internal error - model should be in eval() mode! '
    assert not model_ts.training, 'internal error - converted model should be in eval() mode! '
    models = model, model_ts
    outputs, time_model, outputs_ts, time_model_ts = self.lib.run_models(models
        , inputs)
    Error_stats = self.lib.compute_errors(outputs, outputs_ts)
    self.lib.print_errors(Error_stats)
    print('time of error check of native model: ', time_model, 'seconds')
    print('time of error check of ts model: ', time_model_ts, 'seconds')
    print()
    output_shapes = self.lib.get_tuple_of_dynamic_shapes(outputs)
    output_types = [x.dtype for x in outputs[0]]
    config_filename = os.path.join(model_folder, 'config.pbtxt')
    self.lib.write_config(config_filename, input_shapes, input_types,
        output_shapes, output_types)
