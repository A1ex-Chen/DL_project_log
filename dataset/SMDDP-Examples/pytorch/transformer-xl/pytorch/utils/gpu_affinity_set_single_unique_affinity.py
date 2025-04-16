def set_single_unique_affinity(gpu_id, nproc_per_node):
    devices = [device(i) for i in range(nproc_per_node)]
    socket_affinities = [dev.getCpuAffinity() for dev in devices]
    siblings_list = get_thread_siblings_list()
    siblings_dict = dict(siblings_list)
    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities[idx] = list(set(socket_affinity) - set(
            siblings_dict.values()))
    affinities = []
    assigned = []
    for socket_affinity in socket_affinities:
        for core in socket_affinity:
            if core not in assigned:
                affinities.append([core])
                assigned.append(core)
                break
    os.sched_setaffinity(0, affinities[gpu_id])
