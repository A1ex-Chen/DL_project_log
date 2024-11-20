def __post_init__(self):
    if self.disable_tqdm is None:
        self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN
    self.evaluation_strategy = EvaluationStrategy(self.evaluation_strategy)
    if (self.do_eval is False and self.evaluation_strategy !=
        EvaluationStrategy.NO):
        self.do_eval = True
    if self.eval_steps is None:
        self.eval_steps = self.logging_steps
    if self.load_best_model_at_end and self.metric_for_best_model is None:
        self.metric_for_best_model = 'loss'
    if (self.greater_is_better is None and self.metric_for_best_model is not
        None):
        self.greater_is_better = self.metric_for_best_model not in ['loss',
            'eval_loss']
    if self.run_name is None:
        self.run_name = self.output_dir
    if is_torch_available() and self.device.type != 'cuda' and self.fp16:
        raise ValueError('AMP (`--fp16`) can only be used on CUDA devices.')
