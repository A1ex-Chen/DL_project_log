@classmethod
def from_config(cls, pipeline_type, device_id, batch_size, file_root: str,
    sampler, config_data: dict, config_features: dict, seed, device_type:
    str='gpu', do_resampling: bool=True, num_cpu_threads=multiprocessing.
    cpu_count(), synthetic_seq_len=None, in_mem_file_list=True, pre_sort=
    False, dont_use_mmap=False):
    max_duration = config_data['max_duration']
    sample_rate = config_data['sample_rate']
    silence_threshold = -60 if config_data['trim_silence'] else None
    if do_resampling and config_data['speed_perturbation'] is not None:
        resample_range = [config_data['speed_perturbation']['min_rate'],
            config_data['speed_perturbation']['max_rate']]
    else:
        resample_range = None
    window_size = config_features['window_size']
    window_stride = config_features['window_stride']
    nfeatures = config_features['n_filt']
    nfft = config_features['n_fft']
    dither_coeff = config_features['dither']
    preemph_coeff = 0.97
    return cls(pipeline_type=pipeline_type, device_id=device_id,
        preprocessing_device=device_type, num_threads=num_cpu_threads,
        batch_size=batch_size, file_root=file_root, sampler=sampler,
        sample_rate=sample_rate, resample_range=resample_range, window_size
        =window_size, window_stride=window_stride, nfeatures=nfeatures,
        nfft=nfft, dither_coeff=dither_coeff, silence_threshold=
        silence_threshold, preemph_coeff=preemph_coeff, max_duration=
        max_duration, synthetic_seq_len=synthetic_seq_len, seed=seed,
        in_mem_file_list=in_mem_file_list, pre_sort=pre_sort, dont_use_mmap
        =dont_use_mmap)
