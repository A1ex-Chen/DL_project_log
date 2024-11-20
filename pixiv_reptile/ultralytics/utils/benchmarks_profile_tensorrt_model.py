def profile_tensorrt_model(self, engine_file: str, eps: float=0.001):
    """Profiles the TensorRT model, measuring average run time and standard deviation among runs."""
    if not self.trt or not Path(engine_file).is_file():
        return 0.0, 0.0
    model = YOLO(engine_file)
    input_data = np.random.rand(self.imgsz, self.imgsz, 3).astype(np.float32)
    elapsed = 0.0
    for _ in range(3):
        start_time = time.time()
        for _ in range(self.num_warmup_runs):
            model(input_data, imgsz=self.imgsz, verbose=False)
        elapsed = time.time() - start_time
    num_runs = max(round(self.min_time / (elapsed + eps) * self.
        num_warmup_runs), self.num_timed_runs * 50)
    run_times = []
    for _ in TQDM(range(num_runs), desc=engine_file):
        results = model(input_data, imgsz=self.imgsz, verbose=False)
        run_times.append(results[0].speed['inference'])
    run_times = self.iterative_sigma_clipping(np.array(run_times), sigma=2,
        max_iters=3)
    return np.mean(run_times), np.std(run_times)
