def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f'WORKER {worker_id} seed:', np.random.get_state()[1][0])
