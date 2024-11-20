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
