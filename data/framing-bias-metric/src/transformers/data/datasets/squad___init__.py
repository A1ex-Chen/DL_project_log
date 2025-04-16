def __init__(self, args: SquadDataTrainingArguments, tokenizer:
    PreTrainedTokenizer, limit_length: Optional[int]=None, mode: Union[str,
    Split]=Split.train, is_language_sensitive: Optional[bool]=False,
    cache_dir: Optional[str]=None, dataset_format: Optional[str]='pt'):
    self.args = args
    self.is_language_sensitive = is_language_sensitive
    self.processor = SquadV2Processor(
        ) if args.version_2_with_negative else SquadV1Processor()
    if isinstance(mode, str):
        try:
            mode = Split[mode]
        except KeyError:
            raise KeyError('mode is not a valid split name')
    self.mode = mode
    version_tag = 'v2' if args.version_2_with_negative else 'v1'
    cached_features_file = os.path.join(cache_dir if cache_dir is not None else
        args.data_dir, 'cached_{}_{}_{}_{}'.format(mode.value, tokenizer.
        __class__.__name__, str(args.max_seq_length), version_tag))
    lock_path = cached_features_file + '.lock'
    with FileLock(lock_path):
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            start = time.time()
            self.old_features = torch.load(cached_features_file)
            self.features = self.old_features['features']
            self.dataset = self.old_features.get('dataset', None)
            self.examples = self.old_features.get('examples', None)
            logger.info(
                f'Loading features from cached file {cached_features_file} [took %.3f s]'
                , time.time() - start)
            if self.dataset is None or self.examples is None:
                logger.warn(
                    f'Deleting cached file {cached_features_file} will allow dataset and examples to be cached in future run'
                    )
        else:
            if mode == Split.dev:
                self.examples = self.processor.get_dev_examples(args.data_dir)
            else:
                self.examples = self.processor.get_train_examples(args.data_dir
                    )
            self.features, self.dataset = squad_convert_examples_to_features(
                examples=self.examples, tokenizer=tokenizer, max_seq_length
                =args.max_seq_length, doc_stride=args.doc_stride,
                max_query_length=args.max_query_length, is_training=mode ==
                Split.train, threads=args.threads, return_dataset=
                dataset_format)
            start = time.time()
            torch.save({'features': self.features, 'dataset': self.dataset,
                'examples': self.examples}, cached_features_file)
            logger.info('Saving features into cached file %s [took %.3f s]',
                cached_features_file, time.time() - start)
