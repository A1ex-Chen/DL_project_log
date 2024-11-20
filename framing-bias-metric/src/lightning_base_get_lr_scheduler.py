def get_lr_scheduler(self):
    get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
    scheduler = get_schedule_func(self.opt, num_warmup_steps=self.hparams.
        warmup_steps, num_training_steps=self.total_steps())
    scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
    return scheduler
