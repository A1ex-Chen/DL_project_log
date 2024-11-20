def _measure_memory(self, func: Callable[[], None]) ->[Memory, MemorySummary]:
    try:
        if self.args.trace_memory_line_by_line:
            trace = start_memory_tracing('transformers')
        if self.args.is_tpu:
            raise NotImplementedError(
                'Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with `--no-memory` or `args.memory=False`'
                )
        elif self.args.is_gpu:
            if not is_py3nvml_available():
                logger.warning(
                    "py3nvml not installed, we won't log GPU memory usage. Install py3nvml (pip install py3nvml) to log information about GPU."
                    )
                memory = 'N/A'
            else:
                logger.info(
                    'Measuring total GPU usage on GPU device. Make sure to not have additional processes running on the same GPU.'
                    )
                nvml.nvmlInit()
                func()
                handle = nvml.nvmlDeviceGetHandleByIndex(self.args.device_idx)
                meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                max_bytes_in_use = meminfo.used
                memory = Memory(max_bytes_in_use)
                nvml.nvmlShutdown()
        else:
            memory_bytes = measure_peak_memory_cpu(func)
            memory = Memory(memory_bytes) if isinstance(memory_bytes, int
                ) else memory_bytes
        if self.args.trace_memory_line_by_line:
            summary = stop_memory_tracing(trace)
        else:
            summary = None
        return memory, summary
    except RuntimeError as e:
        self.print_fn("Doesn't fit on GPU. {}".format(e))
        return 'N/A', None
