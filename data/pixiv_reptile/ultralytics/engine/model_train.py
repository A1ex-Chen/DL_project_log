def train(self, trainer=None, **kwargs):
    """
        Trains the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings and configurations. It supports
        training with a custom trainer or the default training approach defined in the method. The method handles
        different scenarios, such as resuming training from a checkpoint, integrating with Ultralytics HUB, and
        updating model and configuration after training.

        When using Ultralytics HUB, if the session already has a loaded model, the method prioritizes HUB training
        arguments and issues a warning if local arguments are provided. It checks for pip updates and combines default
        configurations, method-specific defaults, and user-provided arguments to configure the training process. After
        training, it updates the model and its configurations, and optionally attaches metrics.

        Args:
            trainer (BaseTrainer, optional): An instance of a custom trainer class for training the model. If None, the
                method uses a default trainer. Defaults to None.
            **kwargs (any): Arbitrary keyword arguments representing the training configuration. These arguments are
                used to customize various aspects of the training process.

        Returns:
            (dict | None): Training metrics if available and training is successful; otherwise, None.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            PermissionError: If there is a permission issue with the HUB session.
            ModuleNotFoundError: If the HUB SDK is not installed.
        """
    self._check_is_pytorch_model()
    if hasattr(self.session, 'model') and self.session.model.id:
        if any(kwargs):
            LOGGER.warning(
                'WARNING ⚠️ using HUB training arguments, ignoring local training arguments.'
                )
        kwargs = self.session.train_args
    checks.check_pip_update_available()
    overrides = yaml_load(checks.check_yaml(kwargs['cfg'])) if kwargs.get('cfg'
        ) else self.overrides
    custom = {'data': overrides.get('data') or DEFAULT_CFG_DICT['data'] or
        TASK2DATA[self.task], 'model': self.overrides['model'], 'task':
        self.task}
    args = {**overrides, **custom, **kwargs, 'mode': 'train'}
    if args.get('resume'):
        args['resume'] = self.ckpt_path
    self.trainer = (trainer or self._smart_load('trainer'))(overrides=args,
        _callbacks=self.callbacks)
    if not args.get('resume'):
        self.trainer.model = self.trainer.get_model(weights=self.model if
            self.ckpt else None, cfg=self.model.yaml)
        self.model = self.trainer.model
    self.trainer.hub_session = self.session
    self.trainer.train()
    if RANK in {-1, 0}:
        ckpt = self.trainer.best if self.trainer.best.exists(
            ) else self.trainer.last
        self.model, _ = attempt_load_one_weight(ckpt)
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)
    return self.metrics
