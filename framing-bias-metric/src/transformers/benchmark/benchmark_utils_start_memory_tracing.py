def start_memory_tracing(modules_to_trace: Optional[Union[str, Iterable[str
    ]]]=None, modules_not_to_trace: Optional[Union[str, Iterable[str]]]=
    None, events_to_trace: str='line', gpus_to_trace: Optional[List[int]]=None
    ) ->MemoryTrace:
    """
    Setup line-by-line tracing to record rss mem (RAM) at each line of a module or sub-module. See `./benchmark.py` for
    usage examples. Current memory consumption is returned using psutil and in particular is the RSS memory "Resident
    Set Size” (the non-swapped physical memory the process is using). See
    https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

    Args:

        - `modules_to_trace`: (None, string, list/tuple of string) if None, all events are recorded if string or list
          of strings: only events from the listed module/sub-module will be recorded (e.g. 'fairseq' or
          'transformers.models.gpt2.modeling_gpt2')
        - `modules_not_to_trace`: (None, string, list/tuple of string) if None, no module is avoided if string or list
          of strings: events from the listed module/sub-module will not be recorded (e.g. 'torch')
        - `events_to_trace`: string or list of string of events to be recorded (see official python doc for
          `sys.settrace` for the list of events) default to line
        - `gpus_to_trace`: (optional list, default None) list of GPUs to trace. Default to tracing all GPUs

    Return:

        - `memory_trace` is a list of `UsedMemoryState` for each event (default each line of the traced script).

            - `UsedMemoryState` are named tuples with the following fields:

                - 'frame': a `Frame` namedtuple (see below) storing information on the current tracing frame (current
                  file, location in current file)
                - 'cpu_memory': CPU RSS memory state *before* executing the line
                - 'gpu_memory': GPU used memory *before* executing the line (sum for all GPUs or for only
                  `gpus_to_trace` if provided)

    `Frame` is a namedtuple used by `UsedMemoryState` to list the current frame state. `Frame` has the following
    fields: - 'filename' (string): Name of the file currently executed - 'module' (string): Name of the module
    currently executed - 'line_number' (int): Number of the line currently executed - 'event' (string): Event that
    triggered the tracing (default will be "line") - 'line_text' (string): Text of the line in the python script

    """
    if is_psutil_available():
        process = psutil.Process(os.getpid())
    else:
        logger.warning(
            "Psutil not installed, we won't log CPU memory usage. Install psutil (pip install psutil) to use CPU memory tracing."
            )
        process = None
    if is_py3nvml_available():
        try:
            nvml.nvmlInit()
            devices = list(range(nvml.nvmlDeviceGetCount())
                ) if gpus_to_trace is None else gpus_to_trace
            nvml.nvmlShutdown()
        except (OSError, nvml.NVMLError):
            logger.warning(
                "Error while initializing communication with GPU. We won't perform GPU memory tracing."
                )
            log_gpu = False
        else:
            log_gpu = is_torch_available() or is_tf_available()
    else:
        logger.warning(
            "py3nvml not installed, we won't log GPU memory usage. Install py3nvml (pip install py3nvml) to use GPU memory tracing."
            )
        log_gpu = False
    memory_trace = []

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
                elif isinstance(modules_to_trace, (list, tuple)) and all(m
                     not in name for m in modules_to_trace):
                    return traceit
            if modules_not_to_trace is not None:
                if isinstance(modules_not_to_trace, str
                    ) and modules_not_to_trace in name:
                    return traceit
                elif isinstance(modules_not_to_trace, (list, tuple)) and any(
                    m in name for m in modules_not_to_trace):
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
    sys.settrace(traceit)
    global _is_memory_tracing_enabled
    _is_memory_tracing_enabled = True
    return memory_trace
