@property
def gpu_provider(self):
    return 'CUDAExecutionProvider', {'gpu_mem_limit': '15000000000',
        'arena_extend_strategy': 'kSameAsRequested'}
