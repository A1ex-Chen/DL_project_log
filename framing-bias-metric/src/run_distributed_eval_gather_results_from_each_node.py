def gather_results_from_each_node(num_replicas, save_dir, timeout) ->List[Dict
    [str, List]]:
    start_wait = time.time()
    logger.info('waiting for all nodes to finish')
    json_data = None
    while time.time() - start_wait < timeout:
        json_files = list(save_dir.glob('rank_*.json'))
        if len(json_files) < num_replicas:
            continue
        try:
            json_data = lmap(load_json, json_files)
            return json_data
        except JSONDecodeError:
            continue
    else:
        raise TimeoutError('Rank 0 gave up on waiting for other processes')
