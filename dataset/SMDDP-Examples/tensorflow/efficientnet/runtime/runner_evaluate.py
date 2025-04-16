def evaluate(self):
    if self.validation_builder is not None:
        validation_steps = (self.eval_num_examples // self.
            validation_builder.global_batch_size)
    else:
        validation_steps = None
    metrics_map = get_metrics(self.one_hot)
    metrics = [metrics_map[metric] for metric in self.metrics]
    if self.one_hot:
        loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=
            self.params.label_smoothing)
    else:
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
    self.model.compile(optimizer='rmsprop', loss=loss_obj, metrics=metrics,
        experimental_run_tf_function=False)
    _ = self.load_model_weights(self.params.model_dir)
    eval_callback = custom_callbacks.EvalTimeHistory(batch_size=self.params
        .eval_batch_size, logger=self.logger)
    results = self.model.evaluate(self.validation_dataset, steps=
        validation_steps, callbacks=eval_callback, verbose=1)
    build_stats(None, results, None, eval_callback, self.logger)
