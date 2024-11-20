def total_steps(self) ->int:
    """The number of total training steps that will be run. Used for lr scheduler purposes."""
    num_devices = max(1, self.hparams.gpus)
    effective_batch_size = (self.hparams.train_batch_size * self.hparams.
        accumulate_grad_batches * num_devices)
    return self.dataset_size / effective_batch_size * self.hparams.max_epochs
