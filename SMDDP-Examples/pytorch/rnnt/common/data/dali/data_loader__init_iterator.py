def _init_iterator(self, gpu_id, dataset_path, config_data, config_features,
    json_names: list, tokenizer: list, num_threads, pipeline_type,
    synthetic_seq_len, seed, in_mem_file_list, enable_prefetch, preproc,
    tokenized_transcript=False, min_seq_split_len=-1, pre_sort=False,
    jit_tensor_formation=False, dont_use_mmap=False):
    """
        Returns data iterator. Data underneath this operator is preprocessed within Dali
        """
    if in_mem_file_list:
        assert len(self.sampler.files) > 0 and len(self.sampler.labels
            ) > 0, 'Please run sampler.sample() first'
    else:
        assert self.sampler.file_list_path is not None, 'Please run sampler.sample() first'
    self.dataset_size = self.sampler.get_dataset_size()
    print_once(f'Dataset read by DALI. Number of samples: {self.dataset_size}')
    pipeline = DaliPipeline.from_config(config_data=config_data,
        config_features=config_features, device_id=gpu_id, file_root=
        dataset_path, sampler=self.sampler, device_type=self.device_type,
        batch_size=self.batch_size, pipeline_type=pipeline_type,
        num_cpu_threads=num_threads, synthetic_seq_len=synthetic_seq_len,
        seed=seed, in_mem_file_list=in_mem_file_list, pre_sort=pre_sort,
        dont_use_mmap=dont_use_mmap)
    return DaliRnntIterator([pipeline], transcripts=self.sampler.
        transcripts, tokenizer=tokenizer, batch_size=self.batch_size,
        shard_size=self._shard_size(), pipeline_type=pipeline_type,
        synthetic_text_seq_len=synthetic_seq_len[1] if synthetic_seq_len is not
        None else None, tokenized_transcript=tokenized_transcript,
        enable_prefetch=enable_prefetch, preproc=preproc, min_seq_split_len
        =min_seq_split_len, jit_tensor_formation=jit_tensor_formation)
