def run_profiler(model_provider, input_provider, iteration_provider):
    setup(RANK, WORLD_SIZE)
    model = model_provider()
    inputs = input_provider()
    ddp_model = DDP(model, device_ids=[RANK], bucket_cap_mb=DEFAULT_BUCKET_SIZE
        )
    iteration = iteration_provider(ddp_model)
    start = time.time()
    elapsed = 0
    while elapsed < 30:
        for _ in range(100):
            iteration(*inputs)
        elapsed = time.time() - start
    skip_first = 10
    wait = 5
    warmup = 10
    active = 30
    totalIterations = skip_first + wait + warmup + active
    deepviewSchedule = schedule(skip_first=skip_first, wait=wait, warmup=
        warmup, active=active, repeat=1)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=deepviewSchedule, on_trace_ready=_trace_handler) as p:
        for _ in range(totalIterations):
            iteration(*inputs)
            p.step()
    cleanup()
