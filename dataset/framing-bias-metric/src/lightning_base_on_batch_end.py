def on_batch_end(self, trainer, pl_module):
    lr_scheduler = trainer.lr_schedulers[0]['scheduler']
    lrs = {f'lr_group_{i}': lr for i, lr in enumerate(lr_scheduler.get_lr())}
    pl_module.logger.log_metrics(lrs)
