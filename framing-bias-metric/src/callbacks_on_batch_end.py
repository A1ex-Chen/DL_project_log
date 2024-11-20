def on_batch_end(self, trainer, pl_module):
    lrs = {f'lr_group_{i}': param['lr'] for i, param in enumerate(pl_module
        .trainer.optimizers[0].param_groups)}
    pl_module.logger.log_metrics(lrs)
