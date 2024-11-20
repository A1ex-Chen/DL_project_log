def stop_memory_tracing(memory_trace: Optional[MemoryTrace]=None,
    ignore_released_memory: bool=True) ->Optional[MemorySummary]:
    """
    Stop memory tracing cleanly and return a summary of the memory trace if a trace is given.

    Args:

        `memory_trace` (optional output of start_memory_tracing, default: None):
            memory trace to convert in summary
        `ignore_released_memory` (boolean, default: None):
            if True we only sum memory increase to compute total memory

    Return:

        - None if `memory_trace` is None
        - `MemorySummary` namedtuple otherwise with the fields:

            - `sequential`: a list of `MemoryState` namedtuple (see below) computed from the provided `memory_trace` by
              subtracting the memory after executing each line from the memory before executing said line.
            - `cumulative`: a list of `MemoryState` namedtuple (see below) with cumulative increase in memory for each
              line obtained by summing repeated memory increase for a line if it's executed several times. The list is
              sorted from the frame with the largest memory consumption to the frame with the smallest (can be negative
              if memory is released)
            - `total`: total memory increase during the full tracing as a `Memory` named tuple (see below). Line with
              memory release (negative consumption) are ignored if `ignore_released_memory` is `True` (default).

    `Memory` named tuple have fields

        - `byte` (integer): number of bytes,
        - `string` (string): same as human readable string (ex: "3.5MB")

    `Frame` are namedtuple used to list the current frame state and have the following fields:

        - 'filename' (string): Name of the file currently executed
        - 'module' (string): Name of the module currently executed
        - 'line_number' (int): Number of the line currently executed
        - 'event' (string): Event that triggered the tracing (default will be "line")
        - 'line_text' (string): Text of the line in the python script

    `MemoryState` are namedtuples listing frame + CPU/GPU memory with the following fields:

        - `frame` (`Frame`): the current frame (see above)
        - `cpu`: CPU memory consumed at during the current frame as a `Memory` named tuple
        - `gpu`: GPU memory consumed at during the current frame as a `Memory` named tuple
        - `cpu_gpu`: CPU + GPU memory consumed at during the current frame as a `Memory` named tuple
    """
    global _is_memory_tracing_enabled
    _is_memory_tracing_enabled = False
    if memory_trace is not None and len(memory_trace) > 1:
        memory_diff_trace = []
        memory_curr_trace = []
        cumulative_memory_dict = defaultdict(lambda : [0, 0, 0])
        for (frame, cpu_mem, gpu_mem), (next_frame, next_cpu_mem, next_gpu_mem
            ) in zip(memory_trace[:-1], memory_trace[1:]):
            cpu_mem_inc = next_cpu_mem - cpu_mem
            gpu_mem_inc = next_gpu_mem - gpu_mem
            cpu_gpu_mem_inc = cpu_mem_inc + gpu_mem_inc
            memory_diff_trace.append(MemoryState(frame=frame, cpu=Memory(
                cpu_mem_inc), gpu=Memory(gpu_mem_inc), cpu_gpu=Memory(
                cpu_gpu_mem_inc)))
            memory_curr_trace.append(MemoryState(frame=frame, cpu=Memory(
                next_cpu_mem), gpu=Memory(next_gpu_mem), cpu_gpu=Memory(
                next_gpu_mem + next_cpu_mem)))
            cumulative_memory_dict[frame][0] += cpu_mem_inc
            cumulative_memory_dict[frame][1] += gpu_mem_inc
            cumulative_memory_dict[frame][2] += cpu_gpu_mem_inc
        cumulative_memory = sorted(list(cumulative_memory_dict.items()),
            key=lambda x: x[1][2], reverse=True)
        cumulative_memory = list(MemoryState(frame=frame, cpu=Memory(
            cpu_mem_inc), gpu=Memory(gpu_mem_inc), cpu_gpu=Memory(
            cpu_gpu_mem_inc)) for frame, (cpu_mem_inc, gpu_mem_inc,
            cpu_gpu_mem_inc) in cumulative_memory)
        memory_curr_trace = sorted(memory_curr_trace, key=lambda x: x.
            cpu_gpu.bytes, reverse=True)
        if ignore_released_memory:
            total_memory = sum(max(0, step_trace.cpu_gpu.bytes) for
                step_trace in memory_diff_trace)
        else:
            total_memory = sum(step_trace.cpu_gpu.bytes for step_trace in
                memory_diff_trace)
        total_memory = Memory(total_memory)
        return MemorySummary(sequential=memory_diff_trace, cumulative=
            cumulative_memory, current=memory_curr_trace, total=total_memory)
    return None
