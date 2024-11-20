def __init__(self, args: GlueDataTrainingArguments, tokenizer:
    PreTrainedTokenizerBase, limit_length: Optional[int]=None, mode: Union[
    str, Split]=Split.train, cache_dir: Optional[str]=None):
    warnings.warn(
        'This dataset will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py'
        , FutureWarning)
    self.args = args
    self.processor = glue_processors[args.task_name]()
    self.output_mode = glue_output_modes[args.task_name]
    if isinstance(mode, str):
        try:
            mode = Split[mode]
        except KeyError:
            raise KeyError('mode is not a valid split name')
    cached_features_file = os.path.join(cache_dir if cache_dir is not None else
        args.data_dir, 'cached_{}_{}_{}_{}'.format(mode.value, tokenizer.
        __class__.__name__, str(args.max_seq_length), args.task_name))
    label_list = self.processor.get_labels()
    if args.task_name in ['mnli', 'mnli-mm'
        ] and tokenizer.__class__.__name__ in ('RobertaTokenizer',
        'RobertaTokenizerFast', 'XLMRobertaTokenizer', 'BartTokenizer',
        'BartTokenizerFast'):
        label_list[1], label_list[2] = label_list[2], label_list[1]
    self.label_list = label_list
    lock_path = cached_features_file + '.lock'
    with FileLock(lock_path):
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            start = time.time()
            self.features = torch.load(cached_features_file)
            logger.info(
                f'Loading features from cached file {cached_features_file} [took %.3f s]'
                , time.time() - start)
        else:
            logger.info(
                f'Creating features from dataset file at {args.data_dir}')
            if mode == Split.dev:
                examples = self.processor.get_dev_examples(args.data_dir)
            elif mode == Split.test:
                examples = self.processor.get_test_examples(args.data_dir)
            else:
                examples = self.processor.get_train_examples(args.data_dir)
            if limit_length is not None:
                examples = examples[:limit_length]
            self.features = glue_convert_examples_to_features(examples,
                tokenizer, max_length=args.max_seq_length, label_list=
                label_list, output_mode=self.output_mode)
            start = time.time()
            torch.save(self.features, cached_features_file)
            logger.info('Saving features into cached file %s [took %.3f s]',
                cached_features_file, time.time() - start)
