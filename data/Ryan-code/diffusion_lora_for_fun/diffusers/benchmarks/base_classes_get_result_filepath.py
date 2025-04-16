def get_result_filepath(self, args):
    pipeline_class_name = str(self.pipe.__class__.__name__)
    name = (self.lora_id.replace('/', '_') + '_' + pipeline_class_name +
        f'-bs@{args.batch_size}-steps@{args.num_inference_steps}-mco@{args.model_cpu_offload}-compile@{args.run_compile}.csv'
        )
    filepath = os.path.join(BASE_PATH, name)
    return filepath
