def habitat_compute_threshold(self, runnable, context):
    tracker = habitat.OperationTracker(context.origin_device)
    with tracker.track():
        runnable()
    run_times = []
    trace = tracker.get_tracked_trace()
    for op in trace.operations:
        if op.name in SPECIAL_OPERATIONS:
            continue
        run_times.append(op.forward.run_time_ms)
        if op.backward is not None:
            run_times.append(op.backward.run_time_ms)
    return np.percentile(run_times, context.percentile)
