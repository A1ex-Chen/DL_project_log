def measure_peak_memory_cpu(function: Callable[[], None], interval=0.5,
    device_idx=None) ->int:
    """
    measures peak cpu memory consumption of a given `function` running the function for at least interval seconds and
    at most 20 * interval seconds. This function is heavily inspired by: `memory_usage` of the package
    `memory_profiler`:
    https://github.com/pythonprofilers/memory_profiler/blob/895c4ac7a08020d66ae001e24067da6dcea42451/memory_profiler.py#L239

    Args:

        - `function`: (`callable`): function() -> ... function without any arguments to measure for which to measure
          the peak memory

        - `interval`: (`float`, `optional`, defaults to `0.5`) interval in second for which to measure the memory usage

        - `device_idx`: (`int`, `optional`, defaults to `None`) device id for which to measure gpu usage

    Returns:

        - `max_memory`: (`int`) consumed memory peak in Bytes
    """

    def get_cpu_memory(process_id: int) ->int:
        """
        measures current cpu memory usage of a given `process_id`

        Args:

            - `process_id`: (`int`) process_id for which to measure memory

        Returns

            - `memory`: (`int`) consumed memory in Bytes
        """
        process = psutil.Process(process_id)
        try:
            meminfo_attr = 'memory_info' if hasattr(process, 'memory_info'
                ) else 'get_memory_info'
            memory = getattr(process, meminfo_attr)()[0]
        except psutil.AccessDenied:
            raise ValueError('Error with Psutil.')
        return memory
    if not is_psutil_available():
        logger.warning(
            "Psutil not installed, we won't log CPU memory usage. Install Psutil (pip install psutil) to use CPU memory tracing."
            )
        max_memory = 'N/A'
    else:


        class MemoryMeasureProcess(Process):
            """
            `MemoryMeasureProcess` inherits from `Process` and overwrites its `run()` method. Used to measure the
            memory usage of a process
            """

            def __init__(self, process_id: int, child_connection:
                Connection, interval: float):
                super().__init__()
                self.process_id = process_id
                self.interval = interval
                self.connection = child_connection
                self.num_measurements = 1
                self.mem_usage = get_cpu_memory(self.process_id)

            def run(self):
                self.connection.send(0)
                stop = False
                while True:
                    self.mem_usage = max(self.mem_usage, get_cpu_memory(
                        self.process_id))
                    self.num_measurements += 1
                    if stop:
                        break
                    stop = self.connection.poll(self.interval)
                self.connection.send(self.mem_usage)
                self.connection.send(self.num_measurements)
        while True:
            child_connection, parent_connection = Pipe()
            mem_process = MemoryMeasureProcess(os.getpid(),
                child_connection, interval)
            mem_process.start()
            parent_connection.recv()
            try:
                function()
                parent_connection.send(0)
                max_memory = parent_connection.recv()
                num_measurements = parent_connection.recv()
            except Exception:
                parent = psutil.Process(os.getpid())
                for child in parent.children(recursive=True):
                    os.kill(child.pid, SIGKILL)
                mem_process.join(0)
                raise RuntimeError('Process killed. Error in Process')
            mem_process.join(20 * interval)
            if num_measurements > 4 or interval < 1e-06:
                break
            interval /= 10
        return max_memory
