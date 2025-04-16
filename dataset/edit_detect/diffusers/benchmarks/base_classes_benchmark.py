def benchmark(self, args):
    flush()
    print(
        f'[INFO] {self.pipe.__class__.__name__}: Running benchmark with: {vars(args)}\n'
        )
    time = benchmark_fn(self.run_inference, self.pipe, args)
    memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())
    benchmark_info = BenchmarkInfo(time=time, memory=memory)
    pipeline_class_name = str(self.pipe.__class__.__name__)
    flush()
    csv_dict = generate_csv_dict(pipeline_cls=pipeline_class_name, ckpt=
        self.lora_id, args=args, benchmark_info=benchmark_info)
    filepath = self.get_result_filepath(args)
    write_to_csv(filepath, csv_dict)
    print(f'Logs written to: {filepath}')
    flush()
