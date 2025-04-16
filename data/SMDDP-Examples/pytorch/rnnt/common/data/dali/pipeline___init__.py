def __init__(self, *, pipeline_type, device_id, num_threads, batch_size,
    file_root: str, sampler, sample_rate, resample_range: list, window_size,
    window_stride, nfeatures, nfft, dither_coeff, silence_threshold,
    preemph_coeff, max_duration, seed, preprocessing_device='gpu',
    synthetic_seq_len=None, in_mem_file_list=True, pre_sort=False,
    dont_use_mmap=False):
    super().__init__(batch_size, num_threads, device_id, seed)
    self._dali_init_log(locals())
    if dist.is_initialized() and not sampler.dist_sampler:
        shard_id = dist.get_rank()
        n_shards = dist.get_world_size()
    else:
        shard_id = 0
        n_shards = 1
    self.preprocessing_device = preprocessing_device.lower()
    assert self.preprocessing_device == 'cpu' or self.preprocessing_device == 'gpu', "Incorrect preprocessing device. Please choose either 'cpu' or 'gpu'"
    self.resample_range = resample_range
    train_pipeline = pipeline_type == 'train'
    self.train = train_pipeline
    self.sample_rate = sample_rate
    self.dither_coeff = dither_coeff
    self.nfeatures = nfeatures
    self.max_duration = max_duration
    self.do_remove_silence = True if silence_threshold is not None else False
    if pipeline_type == 'train' and pre_sort:
        pci = PertCoeffIterator(batch_size, sampler.pert_coeff)
    shuffle = train_pipeline and not sampler.is_sampler_random()
    if synthetic_seq_len is None:
        if in_mem_file_list:
            self.read = ops.readers.File(name='Reader', dont_use_mmap=
                dont_use_mmap, pad_last_batch=pipeline_type == 'val',
                device='cpu', file_root=file_root, files=sampler.files,
                labels=sampler.labels, shard_id=shard_id, num_shards=
                n_shards, shuffle_after_epoch=shuffle)
        else:
            self.read = ops.readers.File(name='Reader', dont_use_mmap=
                dont_use_mmap, pad_last_batch=pipeline_type == 'val',
                device='cpu', file_root=file_root, file_list=sampler.
                get_file_list_path(), shard_id=shard_id, num_shards=
                n_shards, shuffle_after_epoch=shuffle)
    else:
        self.read = ops.readers.File(name='Reader', dont_use_mmap=
            dont_use_mmap, pad_last_batch=pipeline_type == 'val', device=
            'cpu', file_root=file_root, file_list=
            '/workspace/rnnt/rnnt_dali.file_list.synth', shard_id=shard_id,
            num_shards=n_shards, shuffle_after_epoch=shuffle)
    if resample_range is not None and not (resample_range[0] == 1 and 
        resample_range[1] == 1):
        if pre_sort:
            self.speed_perturbation_coeffs = ops.ExternalSource(source=pci)
        else:
            self.speed_perturbation_coeffs = ops.random.Uniform(device=
                'cpu', range=resample_range)
    else:
        self.speed_perturbation_coeffs = None
    self.decode = ops.decoders.Audio(device='cpu', sample_rate=self.
        sample_rate if resample_range is None else None, dtype=types.FLOAT,
        downmix=True)
    self.normal_distribution = ops.random.Normal(device=preprocessing_device)
    self.preemph = ops.PreemphasisFilter(device=preprocessing_device,
        preemph_coeff=preemph_coeff)
    self.spectrogram = ops.Spectrogram(device=preprocessing_device, nfft=
        nfft, window_length=window_size * sample_rate, window_step=
        window_stride * sample_rate)
    self.mel_fbank = ops.MelFilterBank(device=preprocessing_device,
        sample_rate=sample_rate, nfilter=self.nfeatures, normalize=True)
    self.log_features = ops.ToDecibels(device=preprocessing_device,
        multiplier=np.log(10), reference=1.0, cutoff_db=math.log(1e-20))
    self.get_shape = ops.Shapes(device=preprocessing_device)
    self.normalize = ops.Normalize(device=preprocessing_device, axes=[1])
    self.pad = ops.Pad(device=preprocessing_device, fill_value=0)
    self.get_nonsilent_region = ops.NonsilentRegion(device='cpu', cutoff_db
        =silence_threshold)
    self.trim_silence = ops.Slice(device='cpu', normalized_anchor=False,
        normalized_shape=False, axes=[0])
    self.to_float = ops.Cast(device='cpu', dtype=types.FLOAT)
    self.synthetic_seq_len = synthetic_seq_len
    if self.synthetic_seq_len is not None:
        self.constant = ops.Constant(device=preprocessing_device, dtype=
            types.FLOAT, fdata=100.0, shape=synthetic_seq_len[0])
