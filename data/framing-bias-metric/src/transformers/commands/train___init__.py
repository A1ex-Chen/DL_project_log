def __init__(self, args: Namespace):
    self.logger = logging.get_logger('transformers-cli/training')
    self.framework = 'tf' if is_tf_available() else 'torch'
    os.makedirs(args.output, exist_ok=True)
    self.output = args.output
    self.column_label = args.column_label
    self.column_text = args.column_text
    self.column_id = args.column_id
    self.logger.info('Loading {} pipeline for {}'.format(args.task, args.model)
        )
    if args.task == 'text_classification':
        self.pipeline = TextClassificationPipeline.from_pretrained(args.model)
    elif args.task == 'token_classification':
        raise NotImplementedError
    elif args.task == 'question_answering':
        raise NotImplementedError
    self.logger.info('Loading dataset from {}'.format(args.train_data))
    self.train_dataset = Processor.create_from_csv(args.train_data,
        column_label=args.column_label, column_text=args.column_text,
        column_id=args.column_id, skip_first_row=args.skip_first_row)
    self.valid_dataset = None
    if args.validation_data:
        self.logger.info('Loading validation dataset from {}'.format(args.
            validation_data))
        self.valid_dataset = Processor.create_from_csv(args.validation_data,
            column_label=args.column_label, column_text=args.column_text,
            column_id=args.column_id, skip_first_row=args.skip_first_row)
    self.validation_split = args.validation_split
    self.train_batch_size = args.train_batch_size
    self.valid_batch_size = args.valid_batch_size
    self.learning_rate = args.learning_rate
    self.adam_epsilon = args.adam_epsilon
