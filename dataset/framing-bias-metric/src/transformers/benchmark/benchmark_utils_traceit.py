def traceit(frame, event, args):
    """
        Tracing method executed before running each line in a module or sub-module Record memory allocated in a list
        with debugging information
        """
    global _is_memory_tracing_enabled
    if not _is_memory_tracing_enabled:
        return traceit
    if events_to_trace is not None:
        if isinstance(events_to_trace, str) and event != events_to_trace:
            return traceit
        elif isinstance(events_to_trace, (list, tuple)
            ) and event not in events_to_trace:
            return traceit
    if '__name__' not in frame.f_globals:
        return traceit
    name = frame.f_globals['__name__']
    if not isinstance(name, str):
        return traceit
    else:
        if modules_to_trace is not None:
            if isinstance(modules_to_trace, str
                ) and modules_to_trace not in name:
                return traceit
            elif isinstance(modules_to_trace, (list, tuple)) and all(m not in
                name for m in modules_to_trace):
                return traceit
        if modules_not_to_trace is not None:
            if isinstance(modules_not_to_trace, str
                ) and modules_not_to_trace in name:
                return traceit
            elif isinstance(modules_not_to_trace, (list, tuple)) and any(m in
                name for m in modules_not_to_trace):
                return traceit
    lineno = frame.f_lineno
    filename = frame.f_globals['__file__']
    if filename.endswith('.pyc') or filename.endswith('.pyo'):
        filename = filename[:-1]
    line = linecache.getline(filename, lineno).rstrip()
    traced_state = Frame(filename, name, lineno, event, line)
    cpu_mem = 0
    if process is not None:
        mem = process.memory_info()
        cpu_mem = mem.rss
    gpu_mem = 0
    if log_gpu:
        if is_torch_available():
            torch_empty_cache()
        if is_tf_available():
            tf_context.context()._clear_caches()
        nvml.nvmlInit()
        for i in devices:
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem += meminfo.used
        nvml.nvmlShutdown()
    mem_state = UsedMemoryState(traced_state, cpu_mem, gpu_mem)
    memory_trace.append(mem_state)
    return traceit
