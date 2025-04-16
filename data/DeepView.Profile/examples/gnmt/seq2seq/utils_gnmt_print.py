def gnmt_print(*args, **kwargs):
    """
    Wrapper for MLPerf compliance logging calls.
    All arguments but 'sync' are passed to mlperf_log.gnmt_print function.
    If 'sync' is set to True then the wrapper will synchronize all distributed
    workers. 'sync' should be set to True for all compliance tags that require
    accurate timing (RUN_START, RUN_STOP etc.)
    """
    if kwargs.pop('sync'):
        barrier()
    if get_rank() == 0:
        kwargs['stack_offset'] = 2
