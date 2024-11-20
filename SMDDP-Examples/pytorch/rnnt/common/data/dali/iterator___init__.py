def __init__(self, dali_pipelines, transcripts, tokenizer, batch_size,
    shard_size, pipeline_type, normalize_transcripts=False,
    synthetic_text_seq_len=None, enable_prefetch=False,
    tokenized_transcript=False, preproc=None, min_seq_split_len=-1,
    jit_tensor_formation=False):
    self.normalize_transcripts = normalize_transcripts
    self.tokenizer = tokenizer
    self.tokenized_transcript = tokenized_transcript
    self.batch_size = batch_size
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    if pipeline_type == 'val':
        self.dali_it = DALIGenericIterator(dali_pipelines, ['audio',
            'label', 'audio_shape'], reader_name='Reader', dynamic_shape=
            True, auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)
    else:
        self.dali_it = DALIGenericIterator(dali_pipelines, ['audio',
            'label', 'audio_shape'], size=shard_size, dynamic_shape=True,
            auto_reset=True)
    self.jit_tensor_formation = jit_tensor_formation
    self.tokenize(transcripts)
    self.synthetic_text_seq_len = synthetic_text_seq_len
    self.enable_prefetch = enable_prefetch
    self.prefetch_stream = torch.cuda.Stream()
    self.preproc = preproc
    self.pipeline_type = pipeline_type
    self.min_seq_split_len = min_seq_split_len
    self.pivot_len_cpu = torch.tensor(0, dtype=torch.int, device='cpu'
        ).pin_memory()
