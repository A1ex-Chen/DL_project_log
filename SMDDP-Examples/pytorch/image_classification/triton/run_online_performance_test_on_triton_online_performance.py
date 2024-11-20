def online_performance(model_name: str, batch_sizes: List[int], result_path:
    str, input_shapes: Optional[List[str]]=None, profiling_data: str=
    'random', triton_instances: int=1, triton_gpu_engine_count: int=1,
    server_url: str='localhost', measurement_window: int=10000,
    shared_memory: bool=False):
    print('\n')
    print(f'==== Dynamic batching analysis start ====')
    print('\n')
    input_shapes = ' '.join(map(lambda shape: f' --shape {shape}',
        input_shapes)) if input_shapes else ''
    print(f'Running performance tests for dynamic batching')
    performance_file = f'triton_performance_dynamic_partial.csv'
    max_batch_size = max(batch_sizes)
    max_total_requests = (2 * max_batch_size * triton_instances *
        triton_gpu_engine_count)
    max_concurrency = min(256, max_total_requests)
    batch_size = max(1, max_total_requests // 256)
    step = max(1, max_concurrency // 32)
    min_concurrency = step
    exec_args = (
        f'-m {model_name}         -x 1         -p {measurement_window}         -v         -i http         -u {server_url}:8000         -b {batch_size}         -f {performance_file}         --concurrency-range {min_concurrency}:{max_concurrency}:{step}         --input-data {profiling_data} {input_shapes}'
        )
    if shared_memory:
        exec_args += ' --shared-memory=cuda'
    result = os.system(f'perf_client {exec_args}')
    if result != 0:
        print(
            f'Failed running performance tests. Perf client failed with exit code {result}'
            )
        sys.exit(1)
    results = list()
    update_performance_data(results=results, performance_file=performance_file)
    results = sort_results(results=results)
    save_results(filename=result_path, data=results)
    show_results(results=results)
    os.remove(performance_file)
    print('Performance results for dynamic batching stored in: {0}'.format(
        result_path))
    print('\n')
    print(f'==== Analysis done ====')
    print('\n')
