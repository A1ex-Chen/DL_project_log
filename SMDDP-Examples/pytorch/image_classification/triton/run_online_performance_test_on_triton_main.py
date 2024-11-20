def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True, help=
        'Name of the model to test')
    parser.add_argument('--input-data', type=str, required=False, default=
        'random', help='Input data to perform profiling.')
    parser.add_argument('--input-shape', action='append', required=False,
        help=
        'Input data shape in form INPUT_NAME:<full_shape_without_batch_axis>.')
    parser.add_argument('--batch-sizes', type=str, required=True, help=
        'List of batch sizes to tests. Comma separated.')
    parser.add_argument('--triton-instances', type=int, default=1, help=
        'Number of Triton Server instances')
    parser.add_argument('--number-of-model-instances', type=int, default=1,
        help='Number of models instances on Triton Server')
    parser.add_argument('--result-path', type=str, required=True, help=
        'Path where result file is going to be stored.')
    parser.add_argument('--server-url', type=str, required=False, default=
        'localhost', help='Url to Triton server')
    parser.add_argument('--measurement-window', required=False, help=
        'Time which perf_analyzer will wait for results', default=10000)
    parser.add_argument('--shared-memory', help=
        'Use shared memory for communication with Triton', action=
        'store_true', default=False)
    args = parser.parse_args()
    warmup(server_url=args.server_url, model_name=args.model_name,
        batch_sizes=_parse_batch_sizes(args.batch_sizes), triton_instances=
        args.triton_instances, triton_gpu_engine_count=args.
        number_of_model_instances, profiling_data=args.input_data,
        input_shapes=args.input_shape, measurement_window=args.
        measurement_window, shared_memory=args.shared_memory)
    online_performance(server_url=args.server_url, model_name=args.
        model_name, batch_sizes=_parse_batch_sizes(args.batch_sizes),
        triton_instances=args.triton_instances, triton_gpu_engine_count=
        args.number_of_model_instances, profiling_data=args.input_data,
        input_shapes=args.input_shape, result_path=args.result_path,
        measurement_window=args.measurement_window, shared_memory=args.
        shared_memory)
