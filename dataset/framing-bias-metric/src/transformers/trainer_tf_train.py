def train(self) ->None:
    """
        Train method to train the model.
        """
    train_ds = self.get_train_tfdataset()
    if self.args.debug:
        tf.summary.trace_on(graph=True, profiler=True)
    self.gradient_accumulator.reset()
    num_update_steps_per_epoch = (self.num_train_examples / self.
        total_train_batch_size)
    approx = math.floor if self.args.dataloader_drop_last else math.ceil
    num_update_steps_per_epoch = approx(num_update_steps_per_epoch)
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    self.steps_per_epoch = num_update_steps_per_epoch
    if self.args.max_steps > 0:
        t_total = self.args.max_steps
        epochs = self.args.max_steps // self.steps_per_epoch + int(self.
            args.max_steps % self.steps_per_epoch > 0)
    else:
        t_total = self.steps_per_epoch * self.args.num_train_epochs
        epochs = self.args.num_train_epochs
    epochs = float(epochs)
    with self.args.strategy.scope():
        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        folder = os.path.join(self.args.output_dir, PREFIX_CHECKPOINT_DIR)
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, folder,
            max_to_keep=self.args.save_total_limit)
        iterations = self.optimizer.iterations
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        if self.model.ckpt_manager.latest_checkpoint:
            logger.info(
                'Checkpoint file %s found and restoring from checkpoint',
                self.model.ckpt_manager.latest_checkpoint)
            ckpt.restore(self.model.ckpt_manager.latest_checkpoint
                ).expect_partial()
            self.global_step = iterations.numpy()
            epochs_trained = self.global_step // self.steps_per_epoch
            steps_trained_in_current_epoch = (self.global_step % self.
                steps_per_epoch)
            logger.info(
                '  Continuing training from checkpoint, will skip to saved global_step'
                )
            logger.info('  Continuing training from epoch %d', epochs_trained)
            logger.info('  Continuing training from global step %d', self.
                global_step)
            logger.info('  Will skip the first %d steps in the first epoch',
                steps_trained_in_current_epoch)
        tf.summary.experimental.set_step(self.global_step)
        with self.tb_writer.as_default():
            tf.summary.text('args', self.args.to_json_string())
        self.tb_writer.flush()
        logger.info('***** Running training *****')
        logger.info('  Num examples = %d', self.num_train_examples)
        logger.info('  Num Epochs = %d', epochs)
        logger.info('  Instantaneous batch size per device = %d', self.args
            .per_device_train_batch_size)
        logger.info(
            '  Total train batch size (w. parallel, distributed & accumulation) = %d'
            , self.total_train_batch_size)
        logger.info('  Gradient Accumulation steps = %d', self.args.
            gradient_accumulation_steps)
        logger.info('  Steps per epoch = %d', self.steps_per_epoch)
        logger.info('  Total optimization steps = %d', t_total)
        self.train_loss = tf.keras.metrics.Sum()
        start_time = datetime.datetime.now()
        for epoch_iter in range(epochs_trained, int(epochs)):
            if self.args.past_index >= 0:
                self._past = None
            for step, batch in enumerate(train_ds):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                self.distributed_training_steps(batch)
                self.global_step = iterations.numpy()
                self.epoch_logging = epoch_iter + (step + 1
                    ) / self.steps_per_epoch
                training_loss = self.train_loss.result() / (step + 1)
                if self.args.debug:
                    logs = {}
                    logs['loss'] = training_loss.numpy()
                    logs['epoch'] = self.epoch_logging
                    self.log(logs)
                if self.global_step == 1 and self.args.debug:
                    with self.tb_writer.as_default():
                        tf.summary.trace_export(name='training', step=self.
                            global_step, profiler_outdir=self.args.logging_dir)
                if (self.args.eval_steps > 0 and self.args.
                    evaluate_strategy == EvaluationStrategy.STEPS and self.
                    global_step % self.args.eval_steps == 0):
                    self.evaluate()
                if (self.args.logging_steps > 0 and self.global_step % self
                    .args.logging_steps == 0 or self.global_step == 1 and
                    self.args.logging_first_step):
                    logs = {}
                    logs['loss'] = training_loss.numpy()
                    logs['learning_rate'] = self.lr_scheduler(self.global_step
                        ).numpy()
                    logs['epoch'] = self.epoch_logging
                    self.log(logs)
                if (self.args.save_steps > 0 and self.global_step % self.
                    args.save_steps == 0):
                    ckpt_save_path = self.model.ckpt_manager.save()
                    logger.info('Saving checkpoint for step {} at {}'.
                        format(self.global_step, ckpt_save_path))
                if self.args.max_steps > 0 and self.global_step >= t_total:
                    break
                if self.global_step % self.steps_per_epoch == 0:
                    break
            self.train_loss.reset_states()
            if (self.args.max_steps > 0 and self.global_step >= self.args.
                max_steps):
                break
        end_time = datetime.datetime.now()
        logger.info('Training took: {}'.format(str(end_time - start_time)))
    if self.args.past_index and hasattr(self, '_past'):
        delattr(self, '_past')
