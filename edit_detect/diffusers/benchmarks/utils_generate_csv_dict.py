def generate_csv_dict(pipeline_cls: str, ckpt: str, args: argparse.
    Namespace, benchmark_info: BenchmarkInfo) ->Dict[str, Union[str, bool,
    float]]:
    """Packs benchmarking data into a dictionary for latter serialization."""
    data_dict = {'pipeline_cls': pipeline_cls, 'ckpt_id': ckpt,
        'batch_size': args.batch_size, 'num_inference_steps': args.
        num_inference_steps, 'model_cpu_offload': args.model_cpu_offload,
        'run_compile': args.run_compile, 'time (secs)': benchmark_info.time,
        'memory (gbs)': benchmark_info.memory, 'actual_gpu_memory (gbs)':
        f'{TOTAL_GPU_MEMORY:.3f}', 'github_sha': GITHUB_SHA}
    return data_dict
