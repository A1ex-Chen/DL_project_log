def time_inference_onnxruntime(onnx_filepath, inputs, n_runs_warmup=5,
    profile_execution=False):
    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = (onnxruntime.GraphOptimizationLevel.
        ORT_ENABLE_ALL)
    opt.intra_op_num_threads = 1
    opt.log_severity_level = 0
    opt.enable_profiling = profile_execution
    sess = onnxruntime.InferenceSession(onnx_filepath, opt)
    sess.set_providers(['TensorrtExecutionProvider',
        'CUDAExecutionProvider', 'CPUExecutionProvider'])
    timings = []
    outs = []
    for i in range(len(inputs[0])):
        start_time = time.time()
        sess_inputs = {sess.get_inputs()[j].name: inputs[j][i].numpy() for
            j in range(len(sess.get_inputs()))}
        out = sess.run(None, sess_inputs)[0]
        if i >= n_runs_warmup:
            timings.append(time.time() - start_time)
        outs.append(out.copy())
    return np.array(timings), outs
