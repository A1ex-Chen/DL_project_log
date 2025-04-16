def _get_lr_scheduler(self, num_training_steps):
    schedule_func = arg_to_scheduler[self.args.lr_scheduler]
    if self.args.lr_scheduler == 'constant':
        scheduler = schedule_func(self.optimizer)
    elif self.args.lr_scheduler == 'constant_w_warmup':
        scheduler = schedule_func(self.optimizer, num_warmup_steps=self.
            args.warmup_steps)
    else:
        scheduler = schedule_func(self.optimizer, num_warmup_steps=self.
            args.warmup_steps, num_training_steps=num_training_steps)
    return scheduler
