def profile_onnx_model(self, onnx_file: str, eps: float=0.001):
    """Profiles an ONNX model by executing it multiple times and returns the mean and standard deviation of run
        times.
        """
    check_requirements('onnxruntime')
    import onnxruntime as ort
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (ort.GraphOptimizationLevel.
        ORT_ENABLE_ALL)
    sess_options.intra_op_num_threads = 8
    sess = ort.InferenceSession(onnx_file, sess_options, providers=[
        'CPUExecutionProvider'])
    input_tensor = sess.get_inputs()[0]
    input_type = input_tensor.type
    dynamic = not all(isinstance(dim, int) and dim >= 0 for dim in
        input_tensor.shape)
    input_shape = (1, 3, self.imgsz, self.imgsz
        ) if dynamic else input_tensor.shape
    if 'float16' in input_type:
        input_dtype = np.float16
    elif 'float' in input_type:
        input_dtype = np.float32
    elif 'double' in input_type:
        input_dtype = np.float64
    elif 'int64' in input_type:
        input_dtype = np.int64
    elif 'int32' in input_type:
        input_dtype = np.int32
    else:
        raise ValueError(f'Unsupported ONNX datatype {input_type}')
    input_data = np.random.rand(*input_shape).astype(input_dtype)
    input_name = input_tensor.name
    output_name = sess.get_outputs()[0].name
    elapsed = 0.0
    for _ in range(3):
        start_time = time.time()
        for _ in range(self.num_warmup_runs):
            sess.run([output_name], {input_name: input_data})
        elapsed = time.time() - start_time
    num_runs = max(round(self.min_time / (elapsed + eps) * self.
        num_warmup_runs), self.num_timed_runs)
    run_times = []
    for _ in TQDM(range(num_runs), desc=onnx_file):
        start_time = time.time()
        sess.run([output_name], {input_name: input_data})
        run_times.append((time.time() - start_time) * 1000)
    run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2,
        max_iters=5)
    return np.mean(run_times), np.std(run_times)
