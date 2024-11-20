@staticmethod
def _hook_rss_memory_post_forward(module, *args, **kwargs):
    try:
        import psutil
    except ImportError:
        raise ImportError(
            'You need to install psutil (pip install psutil) to use memory tracing.'
            )
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    module.mem_rss_post_forward = mem.rss
    mem_rss_diff = module.mem_rss_post_forward - module.mem_rss_pre_forward
    module.mem_rss_diff = mem_rss_diff + (module.mem_rss_diff if hasattr(
        module, 'mem_rss_diff') else 0)
    return None
