def set_single_affinity(gpu_id):
    dev = device(gpu_id)
    affinity = dev.getCpuAffinity()
    os.sched_setaffinity(0, affinity[:1])
