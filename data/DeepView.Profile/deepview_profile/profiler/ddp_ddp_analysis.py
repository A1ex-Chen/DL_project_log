def ddp_analysis(model_provider, input_provider, iteration_provider):
    run_profiler(model_provider, input_provider, iteration_provider)
    path_to_file = os.path.join(os.getcwd(), FILENAME)
    fw_avg_msec, bucket_comp_times = _bucket_comp_times(path_to_file)
    bucket_sizes_arr = get_bucket_sizes(model_provider(), DEFAULT_BUCKET_SIZE)
    expected_max_2gpus = _bucket_expected_max(bucket_comp_times, 2)
    expected_max_4gpus = _bucket_expected_max(bucket_comp_times, 4)
    jsonFormat = {'forward_time_ms': fw_avg_msec, 'bucket_sizes':
        bucket_sizes_arr, 'expected_computation_times': [{'ngpus': 2,
        'expected_max_times': expected_max_2gpus}, {'ngpus': 4,
        'expected_max_times': expected_max_4gpus}]}
    subprocess.run(['rm', '-f', os.path.join(os.getcwd(), FILENAME)])
    return jsonFormat
