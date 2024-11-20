def set_socket_affinity(gpu_id):
    dev = device(gpu_id)
    affinity = dev.getCpuAffinity()
    os.sched_setaffinity(0, affinity)
