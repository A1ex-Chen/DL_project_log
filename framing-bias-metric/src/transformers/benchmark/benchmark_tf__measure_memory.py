def _measure_memory(self, func: Callable[[], None]) ->[Memory, MemorySummary]:
    logger.info(
        'Note that TensorFlow allocates more memory thanit might need to speed up computation.The memory reported here corresponds to the memoryreported by `nvidia-smi`, which can vary dependingon total available memory on the GPU that is used.'
        )
    with self.args.strategy.scope():
        try:
            if self.args.trace_memory_line_by_line:
                assert self.args.eager_mode, '`args.eager_mode` is set to `False`. Make sure to run model in eager mode to measure memory consumption line by line.'
                trace = start_memory_tracing('transformers')
            if self.args.is_tpu:
                raise NotImplementedError(
                    'Memory Benchmarking is currently not implemented for TPU. Please disable memory benchmarking with `args.memory=False`'
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
                    handle = nvml.nvmlDeviceGetHandleByIndex(self.args.
                        device_idx)
                    meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                    max_bytes_in_use = meminfo.used
                    memory = Memory(max_bytes_in_use)
                    nvml.nvmlShutdown()
            elif self.args.trace_memory_line_by_line:
                logger.info(
                    'When enabling line by line tracing, the max peak memory for CPU is inaccurate in TensorFlow.'
                    )
                memory = None
            else:
                memory_bytes = measure_peak_memory_cpu(func)
                memory = Memory(memory_bytes) if isinstance(memory_bytes, int
                    ) else memory_bytes
            if self.args.trace_memory_line_by_line:
                summary = stop_memory_tracing(trace)
                if memory is None:
                    memory = summary.total
            else:
                summary = None
            return memory, summary
        except ResourceExhaustedError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
            return 'N/A', None
