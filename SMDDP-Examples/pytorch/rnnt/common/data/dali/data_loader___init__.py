def __init__(self, gpu_id, dataset_path: str, config_data: dict,
    config_features: dict, json_names: list, tokenizer, batch_size: int,
    sampler, pipeline_type: str, seed, grad_accumulation_steps: int=1,
    num_threads=multiprocessing.cpu_count(), tokenized_transcript=False,
    device_type: str='gpu', synthetic_seq_len=None, in_mem_file_list=True,
    enable_prefetch=False, preproc=None, min_seq_split_len=-1, pre_sort=
    False, jit_tensor_formation=False, dont_use_mmap=False):
    import torch
    self.batch_size = batch_size
    self.grad_accumulation_steps = grad_accumulation_steps
    self.drop_last = pipeline_type == 'train'
    self.device_type = device_type
    self.pipeline_type = self._parse_pipeline_type(pipeline_type)
    self.sampler = sampler
    self._dali_data_iterator = self._init_iterator(gpu_id=gpu_id,
        dataset_path=dataset_path, config_data=config_data, config_features
        =config_features, json_names=json_names, tokenizer=tokenizer,
        num_threads=num_threads, pipeline_type=pipeline_type,
        synthetic_seq_len=synthetic_seq_len, seed=seed, in_mem_file_list=
        in_mem_file_list, tokenized_transcript=tokenized_transcript,
        enable_prefetch=enable_prefetch, preproc=preproc, min_seq_split_len
        =min_seq_split_len, pre_sort=pre_sort, jit_tensor_formation=
        jit_tensor_formation, dont_use_mmap=dont_use_mmap)
