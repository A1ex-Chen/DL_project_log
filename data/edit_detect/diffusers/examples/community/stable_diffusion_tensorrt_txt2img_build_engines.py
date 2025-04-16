def build_engines(models: dict, engine_dir, onnx_dir, onnx_opset,
    opt_image_height, opt_image_width, opt_batch_size=1,
    force_engine_rebuild=False, static_batch=False, static_shape=True,
    enable_preview=False, enable_all_tactics=False, timing_cache=None,
    max_workspace_size=0):
    built_engines = {}
    if not os.path.isdir(onnx_dir):
        os.makedirs(onnx_dir)
    if not os.path.isdir(engine_dir):
        os.makedirs(engine_dir)
    for model_name, model_obj in models.items():
        engine_path = getEnginePath(model_name, engine_dir)
        if force_engine_rebuild or not os.path.exists(engine_path):
            logger.warning('Building Engines...')
            logger.warning('Engine build can take a while to complete')
            onnx_path = getOnnxPath(model_name, onnx_dir, opt=False)
            onnx_opt_path = getOnnxPath(model_name, onnx_dir)
            if force_engine_rebuild or not os.path.exists(onnx_opt_path):
                if force_engine_rebuild or not os.path.exists(onnx_path):
                    logger.warning(f'Exporting model: {onnx_path}')
                    model = model_obj.get_model()
                    with torch.inference_mode(), torch.autocast('cuda'):
                        inputs = model_obj.get_sample_input(opt_batch_size,
                            opt_image_height, opt_image_width)
                        torch.onnx.export(model, inputs, onnx_path,
                            export_params=True, opset_version=onnx_opset,
                            do_constant_folding=True, input_names=model_obj
                            .get_input_names(), output_names=model_obj.
                            get_output_names(), dynamic_axes=model_obj.
                            get_dynamic_axes())
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    logger.warning(f'Found cached model: {onnx_path}')
                if force_engine_rebuild or not os.path.exists(onnx_opt_path):
                    logger.warning(
                        f'Generating optimizing model: {onnx_opt_path}')
                    onnx_opt_graph = model_obj.optimize(onnx.load(onnx_path))
                    onnx.save(onnx_opt_graph, onnx_opt_path)
                else:
                    logger.warning(
                        f'Found cached optimized model: {onnx_opt_path} ')
    for model_name, model_obj in models.items():
        engine_path = getEnginePath(model_name, engine_dir)
        engine = Engine(engine_path)
        onnx_path = getOnnxPath(model_name, onnx_dir, opt=False)
        onnx_opt_path = getOnnxPath(model_name, onnx_dir)
        if force_engine_rebuild or not os.path.exists(engine.engine_path):
            engine.build(onnx_opt_path, fp16=True, input_profile=model_obj.
                get_input_profile(opt_batch_size, opt_image_height,
                opt_image_width, static_batch=static_batch, static_shape=
                static_shape), enable_preview=enable_preview, timing_cache=
                timing_cache, workspace_size=max_workspace_size)
        built_engines[model_name] = engine
    for model_name, model_obj in models.items():
        engine = built_engines[model_name]
        engine.load()
        engine.activate()
    return built_engines
