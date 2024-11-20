def utilization_analysis(model_provider, input_provider, iteration_provider):
    model = model_provider()
    inputs = input_provider()
    iteration = iteration_provider(model)
    skip_first = 2
    wait = 1
    warmup = 1
    active = 1
    totalIterations = skip_first + wait + warmup + active
    deepviewSchedule = schedule(skip_first=skip_first, wait=wait, warmup=
        warmup, active=active, repeat=1)
    start = time.time()
    elapsed = 0
    while elapsed < 30:
        for _ in range(100):
            iteration(*inputs)
        elapsed = time.time() - start
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=deepviewSchedule, on_trace_ready=_trace_handler,
        with_stack=True) as p:
        for _ in range(totalIterations):
            iteration(*inputs)
            p.step()
    utilization = UtilizationProfiler()
    path_to_file = os.path.join(os.getcwd(), FILENAME)
    utilization._deepview_analysis(path_to_file)
    jsonFormat = {'root_node': utilization._convert_node_to_dict(
        utilization._root_node), 'tensor_core_perc': utilization.
        _tensor_core_perc}
    return jsonFormat
