def log(self, logs: Dict[str, float]) ->None:
    """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
    logs['epoch'] = self.epoch_logging
    if self.tb_writer:
        with self.tb_writer.as_default():
            for k, v in logs.items():
                tf.summary.scalar(k, v, step=self.global_step)
        self.tb_writer.flush()
    if is_wandb_available():
        wandb.log(logs, step=self.global_step)
    if is_comet_available():
        experiment = comet_ml.config.get_global_experiment()
        if experiment is not None:
            experiment._log_metrics(logs, step=self.global_step, epoch=self
                .epoch_logging, framework='transformers')
    output = {**logs, **{'step': self.global_step}}
    logger.info(output)
