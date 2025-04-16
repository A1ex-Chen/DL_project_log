def get_config(self):
    scheduler_config = self.scheduler.get_config()
    scheduler_config['initial_learning_rate'] = self.initial_learning_rate
    scheduler_config['warmup_steps'] = self.warmup_steps
    scheduler_config['warmup_type'] = self.warmup_type
    scheduler_config['overlap'] = self.overlap
    return scheduler_config
