def offline_performance(model_name: str, batch_sizes: List[int],
    result_path: str, input_shapes: Optional[List[str]]=None,
    profiling_data: str='random', triton_instances: int=1, server_url: str=
    'localhost', measurement_window: int=10000, shared_memory: bool=False):
    print('\n')
    print(f'==== Static batching analysis start ====')
    print('\n')
    input_shapes = ' '.join(map(lambda shape: f' --shape {shape}',
        input_shapes)) if input_shapes else ''
    results: List[Dict] = list()
    for batch_size in batch_sizes:
        print(f'Running performance tests for batch size: {batch_size}')
        performance_partial_file = (
            f'triton_performance_partial_{batch_size}.csv')
        exec_args = (
            f'-max-threads {triton_instances}            -m {model_name}            -x 1            -c {triton_instances}            -t {triton_instances}            -p {measurement_window}            -v            -i http            -u {server_url}:8000            -b {batch_size}            -f {performance_partial_file}            --input-data {profiling_data} {input_shapes}'
            )
        if shared_memory:
            exec_args += ' --shared-memory=cuda'
        result = os.system(f'perf_client {exec_args}')
        if result != 0:
            print(
                f'Failed running performance tests. Perf client failed with exit code {result}'
                )
            sys.exit(1)
        update_performance_data(results, batch_size, performance_partial_file)
        os.remove(performance_partial_file)
    results = sort_results(results=results)
    save_results(filename=result_path, data=results)
    show_results(results=results)
    print('Performance results for static batching stored in: {0}'.format(
        result_path))
    print('\n')
    print(f'==== Analysis done ====')
    print('\n')
