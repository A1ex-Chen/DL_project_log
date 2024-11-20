def get_first_last_step(filepath: str) ->Tuple[int, int]:
    """
    Input: Pytorch profiler trace
    Output: First and last step of the trace.
    """
    tp = get_perfetto_object(filepath)
    steps_arr = tp.query_dict(
        f"select * from slices where name like '%ProfilerStep#%'")
    first_step = int(steps_arr[0]['name'].split('#')[1])
    last_step = int(steps_arr[-1]['name'].split('#')[1])
    return first_step, last_step
