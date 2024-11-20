def time_inference_tensorrt(onnx_filepath, inputs, trt_floatx=16,
    trt_batchsize=1, trt_workspace=2 << 30, n_runs_warmup=5,
    force_tensorrt_engine_rebuild=True):
    trt_filepath = os.path.splitext(onnx_filepath)[0] + '.trt'
    engine = get_engine(onnx_filepath, trt_filepath, trt_floatx=trt_floatx,
        trt_batchsize=trt_batchsize, trt_workspace=trt_workspace,
        force_rebuild=force_tensorrt_engine_rebuild)
    context = engine.create_execution_context()
    in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
    timings = []
    pointers = [int(in_) for in_ in in_gpu] + [int(out_gpu)]
    outs = []
    for i in range(len(inputs[0])):
        start_time = time.time()
        cuda.memcpy_htod(in_gpu[0], inputs[0][i].numpy())
        if len(inputs) == 2:
            cuda.memcpy_htod(in_gpu[1], inputs[1][i].numpy())
        context.execute(1, pointers)
        cuda.memcpy_dtoh(out_cpu, out_gpu)
        if i >= n_runs_warmup:
            timings.append(time.time() - start_time)
        outs.append(out_cpu.copy())
    return np.array(timings), outs
