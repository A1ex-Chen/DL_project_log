def __init__(self, model_path, dict_path=
    'ChEF/models/kosmos2/data/dict.txt', tokenizer_path=
    'ChEF/models/kosmos2/data/sentencepiece.bpe.model', if_grounding=True,
    device='cuda', **kwargs):
    print('Kosmos only supports single GPU evaluation.')
    parser = options.get_interactive_generation_parser()
    input_args = ['--local_rank=0', 'None', '--task', 'generation_obj',
        '--path', model_path, '--dict-path', dict_path,
        '--required-batch-size-multiple', '1', '--remove-bpe=sentencepiece',
        '--max-len-b', '500', '--add-bos-token', '--beam', '1',
        '--buffer-size', '1', '--image-feature-length', '64',
        '--locate-special-token', '1', '--batch-size', '1', '--nbest', '1',
        '--no-repeat-ngram-size', '3', '--location-bin-size', '32']
    args = options.parse_args_and_arch(parser, input_args=input_args)
    cfg = convert_namespace_to_omegaconf(args)
    cfg['common_eval']['model_overrides'
        ] = "{'visual_pretrained': '', 'dict_path':'" + dict_path + "'}"
    task = tasks.setup_task(cfg.task)
    self.task = task
    print('cfg.common_eval.path', cfg.common_eval.path)
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    models, _model_args = checkpoint_utils.load_model_ensemble(utils.
        split_paths(cfg.common_eval.path), arg_overrides=overrides, task=
        task, suffix=cfg.checkpoint.checkpoint_suffix, strict=cfg.
        checkpoint.checkpoint_shard_count == 1, num_shards=cfg.checkpoint.
        checkpoint_shard_count)
    self.model = models[0]
    self.move_to_device(cfg, device)
    self.tokenizer = spm.SentencePieceProcessor()
    self.tokenizer.Load(tokenizer_path)
    self.special_tokens = [self.task.source_dictionary[idx] for idx in
        range(self.tokenizer.vocab_size(), len(self.task.source_dictionary))]
    cfg.generation.sampling = False
    cfg.generation.sampling_topp = -1.0
    cfg.generation.temperature = 1.0
    cfg.generation.beam = 1
    cfg.generation.max_len_a = 1
    self.generator = self.task.build_generator([self.model], cfg.generation,
        extra_gen_cls_kwargs=dict(ppl=False))
    self.cfg = cfg
    self.if_grounding = if_grounding
