def __init__(self, model: TFPreTrainedModel, args: TFTrainingArguments,
    train_dataset: Optional[tf.data.Dataset]=None, eval_dataset: Optional[
    tf.data.Dataset]=None, compute_metrics: Optional[Callable[[
    EvalPrediction], Dict]]=None, tb_writer: Optional[tf.summary.
    SummaryWriter]=None, optimizers: Tuple[tf.keras.optimizers.Optimizer,
    tf.keras.optimizers.schedules.LearningRateSchedule]=(None, None)):
    assert parse(tf.__version__).release >= (2, 2, 0
        ), 'You need to run the TensorFlow trainer with at least the version 2.2.0, your version is %r ' % tf.__version__
    self.model = model
    self.args = args
    self.train_dataset = train_dataset
    self.eval_dataset = eval_dataset
    self.compute_metrics = compute_metrics
    self.optimizer, self.lr_scheduler = optimizers
    self.gradient_accumulator = GradientAccumulator()
    self.global_step = 0
    self.epoch_logging = 0
    if tb_writer is not None:
        self.tb_writer = tb_writer
    else:
        self.tb_writer = tf.summary.create_file_writer(self.args.logging_dir)
    if is_wandb_available():
        self.setup_wandb()
    elif os.environ.get('WANDB_DISABLED') != 'true':
        logger.info(
            'You are instantiating a Trainer but W&B is not installed. To use wandb logging, run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface.'
            )
    if is_comet_available():
        self.setup_comet()
    elif os.environ.get('COMET_MODE') != 'DISABLED':
        logger.info(
            'To use comet_ml logging, run `pip/conda install comet_ml` see https://www.comet.ml/docs/python-sdk/huggingface/'
            )
    set_seed(self.args.seed)
