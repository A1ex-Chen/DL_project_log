def set_socket_unique_affinity(gpu_id, nproc_per_node, mode):
    device_ids = [device(i) for i in range(nproc_per_node)]
    socket_affinities = [dev.getCpuAffinity() for dev in device_ids]
    siblings_list = get_thread_siblings_list()
    siblings_dict = dict(siblings_list)
    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities[idx] = list(set(socket_affinity) - set(
            siblings_dict.values()))
    socket_affinities_to_device_ids = collections.defaultdict(list)
    for idx, socket_affinity in enumerate(socket_affinities):
        socket_affinities_to_device_ids[tuple(socket_affinity)].append(idx)
    for socket_affinity, device_ids in socket_affinities_to_device_ids.items():
        devices_per_group = len(device_ids)
        cores_per_device = len(socket_affinity) // devices_per_group
        for group_id, device_id in enumerate(device_ids):
            if device_id == gpu_id:
                if mode == 'interleaved':
                    affinity = list(socket_affinity[group_id::
                        devices_per_group])
                elif mode == 'continuous':
                    affinity = list(socket_affinity[group_id *
                        cores_per_device:(group_id + 1) * cores_per_device])
                else:
                    raise RuntimeError(
                        'Unknown set_socket_unique_affinity mode')
                affinity += [siblings_dict[aff] for aff in affinity if aff in
                    siblings_dict]
                os.sched_setaffinity(0, affinity)
