def train(self):
    train_epochs = self.params.max_epochs
    train_steps = (self.params.steps_per_epoch if self.params.
        steps_per_epoch is not None else self.train_num_examples // self.
        train_builder.global_batch_size)
    if self.validation_builder is not None:
        validation_steps = (self.eval_num_examples // self.
            validation_builder.global_batch_size)
    else:
        validation_steps = None
    learning_rate = optimizer_factory.build_learning_rate(params=
        get_learning_rate_params(name=self.params.lr_decay, initial_lr=self
        .params.lr_init, decay_epochs=self.params.lr_decay_epochs,
        decay_rate=self.params.lr_decay_rate, warmup_epochs=self.params.
        lr_warmup_epochs), batch_size=self.train_builder.global_batch_size,
        train_steps=train_steps, max_epochs=train_epochs)
    optimizer = optimizer_factory.build_optimizer(optimizer_name=self.
        params.optimizer, base_learning_rate=learning_rate, params=
        get_optimizer_params(name=self.params.optimizer, decay=self.params.
        decay, epsilon=self.params.epsilon, momentum=self.params.momentum,
        moving_average_decay=self.params.moving_average_decay, nesterov=
        self.params.nesterov, beta_1=self.params.beta_1, beta_2=self.params
        .beta_2))
    metrics_map = get_metrics(self.one_hot)
    metrics = [metrics_map[metric] for metric in self.metrics]
    optimizer = sdp_keras.DistributedOptimizer(optimizer, compression=sdp.
        Compression.fp16)
    if self.one_hot:
        loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=
            self.params.label_smoothing)
    else:
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    self.model.compile(optimizer=optimizer, loss=loss_obj, metrics=metrics,
        experimental_run_tf_function=False)
    initial_epoch = 0
    if self.params.resume_checkpoint:
        initial_epoch = self.resume_from_checkpoint(model_dir=self.params.
            model_dir, train_steps=train_steps)
    callbacks = [sdp_callbacks.BroadcastGlobalVariablesCallback(0)]
    callbacks += custom_callbacks.get_callbacks(model_checkpoint=self.
        params.enable_checkpoint_and_export, include_tensorboard=self.
        params.enable_tensorboard, time_history=self.params.time_history,
        track_lr=True, write_model_weights=self.params.write_model_weights,
        initial_step=initial_epoch * train_steps, batch_size=self.
        train_builder.global_batch_size, log_steps=self.params.log_steps,
        model_dir=self.params.model_dir, save_checkpoint_freq=train_steps *
        self.params.save_checkpoint_freq, logger=self.logger)
    if 'eval' not in self.params.mode:
        validation_kwargs = {}
    else:
        validation_kwargs = {'validation_data': self.validation_dataset,
            'validation_steps': validation_steps, 'validation_freq': self.
            params.num_epochs_between_eval}
    history = self.model.fit(self.train_dataset, epochs=train_epochs,
        steps_per_epoch=train_steps, initial_epoch=initial_epoch, callbacks
        =callbacks, verbose=2, **validation_kwargs)
    validation_output = None
    eval_callback = None
    if not self.params.skip_eval and self.validation_builder is not None:
        eval_callback = custom_callbacks.EvalTimeHistory(batch_size=self.
            params.eval_batch_size, logger=self.logger)
        worker_validation_output = self.model.evaluate(self.
            validation_dataset, steps=validation_steps, callbacks=
            eval_callback, verbose=2)
        validation_output = list(sdp.oob_allreduce(worker_validation_output))
    build_stats(history, validation_output, callbacks, eval_callback, self.
        logger)
