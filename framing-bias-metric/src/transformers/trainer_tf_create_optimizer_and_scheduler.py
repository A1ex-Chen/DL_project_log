def create_optimizer_and_scheduler(self, num_training_steps: int):
    """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        TFTrainer's init through :obj:`optimizers`, or subclass and override this method.
        """
    if not self.optimizer and not self.lr_scheduler:
        self.optimizer, self.lr_scheduler = create_optimizer(self.args.
            learning_rate, num_training_steps, self.args.warmup_steps,
            adam_beta1=self.args.adam_beta1, adam_beta2=self.args.
            adam_beta2, adam_epsilon=self.args.adam_epsilon,
            weight_decay_rate=self.args.weight_decay, power=self.args.
            poly_power)
