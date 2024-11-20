def main():
    parser = argparse.ArgumentParser(description=
        'Create Triton model repository and model configuration',
        allow_abbrev=False)
    parser.add_argument('--model-repository', required=True, help=
        'Path to Triton model repository.')
    parser.add_argument('--model-path', required=True, help=
        'Path to model to configure')
    parser.add_argument('--model-format', required=True, choices=
        _available_enum_values(Format), help='Format of model to deploy')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--model-version', default='1', help=
        'Version of model (default 1)')
    parser.add_argument('--max-batch-size', type=int, default=32, help=
        'Maximum batch size allowed for inference. A max_batch_size value of 0 indicates that batching is not allowed for the model'
        )
    parser.add_argument('--precision', type=str, default=Precision.FP16.
        value, choices=_available_enum_values(Precision), help=
        'Model precision (parameter used only by Tensorflow backend with TensorRT optimization)'
        )
    parser.add_argument('--server-url', type=str, default=
        'grpc://localhost:8001', help=
        'Inference server URL in format protocol://host[:port] (default grpc://localhost:8001)'
        )
    parser.add_argument('--load-model', choices=['none', 'poll', 'explicit'
        ], help=
        'Loading model while Triton Server is in given model control mode')
    parser.add_argument('--timeout', default=120, help=
        'Timeout in seconds to wait till model load (default=120)', type=int)
    parser.add_argument('--backend-accelerator', type=str, choices=
        _available_enum_values(Accelerator), default=Accelerator.TRT.value,
        help='Select Backend Accelerator used to serve model')
    parser.add_argument('--number-of-model-instances', type=int, default=1,
        help='Number of model instances per GPU')
    parser.add_argument('--preferred-batch-sizes', type=int, nargs='*',
        help=
        'Batch sizes that the dynamic batcher should attempt to create. In case --max-queue-delay-us is set and this parameter is not, default value will be --max-batch-size'
        )
    parser.add_argument('--max-queue-delay-us', type=int, default=0, help=
        'Max delay time which dynamic batcher shall wait to form a batch (default 0)'
        )
    parser.add_argument('--capture-cuda-graph', type=int, default=0, help=
        'Use cuda capture graph (used only by TensorRT platform)')
    parser.add_argument('-v', '--verbose', help='Provide verbose logs',
        type=str2bool, default=False)
    args = parser.parse_args()
    set_logger(verbose=args.verbose)
    log_dict('args', vars(args))
    config = ModelConfig.create(model_path=args.model_path, model_name=args
        .model_name, model_version=args.model_version, model_format=args.
        model_format, precision=args.precision, max_batch_size=args.
        max_batch_size, accelerator=args.backend_accelerator,
        gpu_engine_count=args.number_of_model_instances,
        preferred_batch_sizes=args.preferred_batch_sizes or [],
        max_queue_delay_us=args.max_queue_delay_us, capture_cuda_graph=args
        .capture_cuda_graph)
    model_store = TritonModelStore(args.model_repository)
    model_store.deploy_model(model_config=config, model_path=args.model_path)
    if args.load_model != 'none':
        client = TritonClient(server_url=args.server_url, verbose=args.verbose)
        client.wait_for_server_ready(timeout=args.timeout)
        if args.load_model == 'explicit':
            client.load_model(model_name=args.model_name)
        if args.load_model == 'poll':
            time.sleep(15)
        client.wait_for_model(model_name=args.model_name, model_version=
            args.model_version, timeout_s=args.timeout)
