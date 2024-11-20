def set_affinity(gpu_id, nproc_per_node, mode='socket'):
    if mode == 'socket':
        set_socket_affinity(gpu_id)
    elif mode == 'single':
        set_single_affinity(gpu_id)
    elif mode == 'single_unique':
        set_single_unique_affinity(gpu_id, nproc_per_node)
    elif mode == 'socket_unique_interleaved':
        set_socket_unique_affinity(gpu_id, nproc_per_node, 'interleaved')
    elif mode == 'socket_unique_continuous':
        set_socket_unique_affinity(gpu_id, nproc_per_node, 'continuous')
    else:
        raise RuntimeError('Unknown affinity mode')
    affinity = os.sched_getaffinity(0)
    return affinity
