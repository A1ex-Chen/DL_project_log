def step(self, cur_epoch, cur_step):
    total_cur_step = cur_epoch * self.iters_per_epoch + cur_step
    if total_cur_step < self.warmup_steps:
        warmup_lr_schedule(step=cur_step, optimizer=self.optimizer,
            max_step=self.warmup_steps, init_lr=self.warmup_start_lr,
            max_lr=self.init_lr)
    else:
        cosine_lr_schedule(epoch=total_cur_step, optimizer=self.optimizer,
            max_epoch=self.max_epoch * self.iters_per_epoch, init_lr=self.
            init_lr, min_lr=self.min_lr)
