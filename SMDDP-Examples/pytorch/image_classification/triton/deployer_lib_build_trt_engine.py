def build_trt_engine(self, model_file, shapes):
    """ takes a path to an onnx file, and shape information, returns a trt engine
            :: model_file :: path to an onnx model
            :: shapes :: dictionary containing min shape, max shape, opt shape for the trt engine
        """
    import tensorrt as trt
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.fp16_mode = self.args.trt_fp16
    builder.max_batch_size = self.args.triton_max_batch_size
    config = builder.create_builder_config()
    config.max_workspace_size = self.args.max_workspace_size
    if self.args.trt_fp16:
        config.flags |= 1 << int(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for s in shapes:
        profile.set_shape(s['name'], min=s['min'], opt=s['opt'], max=s['max'])
    config.add_optimization_profile(profile)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parser.parse(model.read())
            for i in range(parser.num_errors):
                e = parser.get_error(i)
                print('||||e', e)
            engine = builder.build_engine(network, config=config)
    return engine
