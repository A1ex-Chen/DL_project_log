def __init__(self, model: Union[PreTrainedModel, torch.nn.Module]=None,
    args: TrainingArguments=None, data_collator: Optional[DataCollator]=
    None, train_dataset: Optional[Dataset]=None, eval_dataset: Optional[
    Dataset]=None, tokenizer: Optional['PreTrainedTokenizerBase']=None,
    model_init: Callable[[], PreTrainedModel]=None, compute_metrics:
    Optional[Callable[[EvalPrediction], Dict]]=None, callbacks: Optional[
    List[TrainerCallback]]=None, optimizers: Tuple[torch.optim.Optimizer,
    torch.optim.lr_scheduler.LambdaLR]=(None, None)):
    if args is None:
        logger.info(
            'No `TrainingArguments` passed, using the current path as `output_dir`.'
            )
        args = TrainingArguments('tmp_trainer')
    self.args = args
    set_seed(self.args.seed)
    assert model is not None or model_init is not None, 'You must provide a model to use `Trainer`, either by using the `model` argument or the `model_init` argument.'
    self.model_init = model_init
    self.hp_name = None
    if model is None and model_init is not None:
        model = self.call_model_init()
    if not self.args.model_parallel:
        self.model = model.to(args.device) if model is not None else None
    else:
        self.model = model if model is not None else None
    default_collator = (default_data_collator if tokenizer is None else
        DataCollatorWithPadding(tokenizer))
    self.data_collator = (data_collator if data_collator is not None else
        default_collator)
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.tokenizer = tokenizer
    self.compute_metrics = compute_metrics
    self.optimizer, self.lr_scheduler = optimizers
    if model_init is not None and (self.optimizer is not None or self.
        lr_scheduler is not None):
        raise RuntimeError(
            'Passing a `model_init` is incompatible with providing the `optimizers` argument.You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method.'
            )
    callbacks = (DEFAULT_CALLBACKS if callbacks is None else 
        DEFAULT_CALLBACKS + callbacks)
    self.callback_handler = CallbackHandler(callbacks, self.model, self.
        optimizer, self.lr_scheduler)
    self.add_callback(PrinterCallback if self.args.disable_tqdm else
        DEFAULT_PROGRESS_CALLBACK)
    self._loggers_initialized = False
    if self.is_world_process_zero():
        os.makedirs(self.args.output_dir, exist_ok=True)
    if is_torch_tpu_available() and isinstance(self.model, PreTrainedModel):
        self.model.config.xla_device = True
    if not callable(self.data_collator) and callable(getattr(self.
        data_collator, 'collate_batch', None)):
        raise ValueError(
            'The `data_collator` should be a simple callable (function, class with `__call__`).'
            )
    if args.max_steps > 0:
        logger.info(
            'max_steps is given, it will override any value given in num_train_epochs'
            )
    if train_dataset is not None and not isinstance(train_dataset,
        collections.abc.Sized) and args.max_steps <= 0:
        raise ValueError(
            'train_dataset does not implement __len__, max_steps has to be specified'
            )
    if eval_dataset is not None and not isinstance(eval_dataset,
        collections.abc.Sized):
        raise ValueError('eval_dataset must implement __len__')
    if is_datasets_available():
        if isinstance(train_dataset, datasets.Dataset):
            self._remove_unused_columns(self.train_dataset, description=
                'training')
        if isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(self.eval_dataset, description=
                'evaluation')
    self.state = TrainerState()
    self.control = TrainerControl()
    self._total_flos = None
    if self.args.fp16 and _use_native_amp:
        self.scaler = torch.cuda.amp.GradScaler()
    self.hp_search_backend = None
    self.use_tune_checkpoints = False
    default_label_names = ['start_positions', 'end_positions'] if type(self
        .model) in MODEL_FOR_QUESTION_ANSWERING_MAPPING.values() else ['labels'
        ]
    self.label_names = (default_label_names if self.args.label_names is
        None else self.args.label_names)
    self.control = self.callback_handler.on_init_end(self.args, self.state,
        self.control)
