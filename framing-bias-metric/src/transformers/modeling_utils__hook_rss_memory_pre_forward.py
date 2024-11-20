@staticmethod
def _hook_rss_memory_pre_forward(module, *args, **kwargs):
    try:
        import psutil
    except ImportError:
        raise ImportError(
            'You need to install psutil (pip install psutil) to use memory tracing.'
            )
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    module.mem_rss_pre_forward = mem.rss
    return None
