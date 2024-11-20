def train(self, model_path: Optional[str]=None, trial: Union['optuna.Trial',
    Dict[str, Any]]=None):
    """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
    self._hp_search_setup(trial)
    if self.model_init is not None:
        set_seed(self.args.seed)
        model = self.call_model_init(trial)
        if not self.args.model_parallel:
            self.model = model.to(self.args.device)
        self.optimizer, self.lr_scheduler = None, None
    train_dataset_is_sized = isinstance(self.train_dataset, collections.abc
        .Sized)
    train_dataloader = self.get_train_dataloader()
    if train_dataset_is_sized:
        num_update_steps_per_epoch = len(train_dataloader
            ) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.args.max_steps > 0:
            max_steps = self.args.max_steps
            num_train_epochs = (self.args.max_steps //
                num_update_steps_per_epoch + int(self.args.max_steps %
                num_update_steps_per_epoch > 0))
        else:
            max_steps = math.ceil(self.args.num_train_epochs *
                num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.args.num_train_epochs)
    else:
        max_steps = self.args.max_steps
        num_train_epochs = 1
        num_update_steps_per_epoch = max_steps
    self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    self.state = TrainerState()
    self.state.is_hyper_param_search = trial is not None
    self._load_optimizer_and_scheduler(model_path)
    model = self.model
    if self.args.fp16 and _use_apex:
        if not is_apex_available():
            raise ImportError(
                'Please install apex from https://www.github.com/nvidia/apex to use fp16 training.'
                )
        model, self.optimizer = amp.initialize(model, self.optimizer,
            opt_level=self.args.fp16_opt_level)
    if self.args.n_gpu > 1 and not self.args.model_parallel:
        model = torch.nn.DataParallel(model)
    if self.args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids
            =[self.args.local_rank], output_device=self.args.local_rank,
            find_unused_parameters=not getattr(model.config,
            'gradient_checkpointing', False) if isinstance(model,
            PreTrainedModel) else True)
    if is_torch_tpu_available():
        total_train_batch_size = (self.args.train_batch_size * xm.
            xrt_world_size())
    else:
        total_train_batch_size = (self.args.train_batch_size * self.args.
            gradient_accumulation_steps * (torch.distributed.get_world_size
            () if self.args.local_rank != -1 else 1))
    num_examples = (self.num_examples(train_dataloader) if
        train_dataset_is_sized else total_train_batch_size * self.args.
        max_steps)
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {num_examples}')
    logger.info(f'  Num Epochs = {num_train_epochs}')
    logger.info(
        f'  Instantaneous batch size per device = {self.args.per_device_train_batch_size}'
        )
    logger.info(
        f'  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}'
        )
    logger.info(
        f'  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}'
        )
    logger.info(f'  Total optimization steps = {max_steps}')
    self.state.epoch = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if model_path and os.path.isfile(os.path.join(model_path,
        'trainer_state.json')):
        self.state = TrainerState.load_from_json(os.path.join(model_path,
            'trainer_state.json'))
        epochs_trained = self.state.global_step // num_update_steps_per_epoch
        if not self.args.ignore_data_skip:
            steps_trained_in_current_epoch = (self.state.global_step %
                num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= (self.args.
                gradient_accumulation_steps)
        else:
            steps_trained_in_current_epoch = 0
        logger.info(
            '  Continuing training from checkpoint, will skip to saved global_step'
            )
        logger.info(f'  Continuing training from epoch {epochs_trained}')
        logger.info(
            f'  Continuing training from global step {self.state.global_step}')
        if not self.args.ignore_data_skip:
            logger.info(
                f'  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} batches in the first epoch.'
                )
    self.callback_handler.model = self.model
    self.callback_handler.optimizer = self.optimizer
    self.callback_handler.lr_scheduler = self.lr_scheduler
    self.callback_handler.train_dataloader = train_dataloader
    self.state.trial_name = self.hp_name(trial
        ) if self.hp_name is not None else None
    self.state.trial_params = hp_params(trial) if trial is not None else None
    self.state.max_steps = max_steps
    self.state.num_train_epochs = num_train_epochs
    self.state.is_local_process_zero = self.is_local_process_zero()
    self.state.is_world_process_zero = self.is_world_process_zero()
    tr_loss = torch.tensor(0.0).to(self.args.device)
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = 0
    self._total_flos = self.state.total_flos
    model.zero_grad()
    self.control = self.callback_handler.on_train_begin(self.args, self.
        state, self.control)
    if not self.args.ignore_data_skip:
        for epoch in range(epochs_trained):
            for _ in train_dataloader:
                break
    writer = SummaryWriter()
    for epoch in range(epochs_trained, num_train_epochs):
        if isinstance(train_dataloader, DataLoader) and isinstance(
            train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        if is_torch_tpu_available():
            parallel_loader = pl.ParallelLoader(train_dataloader, [self.
                args.device]).per_device_loader(self.args.device)
            epoch_iterator = parallel_loader
        else:
            epoch_iterator = train_dataloader
        if self.args.past_index >= 0:
            self._past = None
        steps_in_epoch = len(epoch_iterator
            ) if train_dataset_is_sized else self.args.max_steps
        self.control = self.callback_handler.on_epoch_begin(self.args, self
            .state, self.control)
        epoch_pbar = tqdm(enumerate(epoch_iterator))
        for step, inputs in epoch_pbar:
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.control = self.callback_handler.on_step_begin(self.
                    args, self.state, self.control)
            if ((step + 1) % self.args.gradient_accumulation_steps != 0 and
                self.args.local_rank != -1 and _use_ddp_no_sync):
                with model.no_sync():
                    tr_loss += self.training_step(model, inputs)
            else:
                tr_loss += self.training_step(model, inputs)
            self._total_flos += self.floating_point_ops(inputs)
            log = 'LOSS: {:.3f}'.format(tr_loss / step)
            epoch_pbar.set_description(log)
            if ((step + 1) % self.args.gradient_accumulation_steps == 0 or 
                steps_in_epoch <= self.args.gradient_accumulation_steps and
                step + 1 == steps_in_epoch):
                writer.add_scalar('loss', tr_loss.item(), step)
                if self.args.fp16 and _use_native_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self
                        .args.max_grad_norm)
                elif self.args.fp16 and _use_apex:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.
                        optimizer), self.args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self
                        .args.max_grad_norm)
                if is_torch_tpu_available():
                    xm.optimizer_step(self.optimizer)
                elif self.args.fp16 and _use_native_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.lr_scheduler.step()
                model.zero_grad()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(self.args,
                    self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)
            if (self.control.should_epoch_stop or self.control.
                should_training_stop):
                break
        self.control = self.callback_handler.on_epoch_end(self.args, self.
            state, self.control)
        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)
        if self.args.tpu_metrics_debug or self.args.debug:
            if is_torch_tpu_available():
                xm.master_print(met.metrics_report())
            else:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU configured. Check your training configuration if this is unexpected."
                    )
        if self.control.should_training_stop:
            break
    if self.args.past_index and hasattr(self, '_past'):
        delattr(self, '_past')
    logger.info(
        """

Training completed. Do not forget to share your model on huggingface.co/models =)

"""
        )
    if (self.args.load_best_model_at_end and self.state.
        best_model_checkpoint is not None):
        logger.info(
            f'Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).'
            )
        if isinstance(self.model, PreTrainedModel):
            self.model = self.model.from_pretrained(self.state.
                best_model_checkpoint)
            if not self.args.model_parallel:
                self.model = self.model.to(self.args.device)
        else:
            state_dict = torch.load(os.path.join(self.state.
                best_model_checkpoint, WEIGHTS_NAME))
            self.model.load_state_dict(state_dict)
    if self._total_flos is not None:
        self.store_flos()
        self.log({'total_flos': self.state.total_flos})
    self.control = self.callback_handler.on_train_end(self.args, self.state,
        self.control)
    self._total_loss_scalar += tr_loss.item()
    return TrainOutput(self.state.global_step, self._total_loss_scalar /
        self.state.global_step)
