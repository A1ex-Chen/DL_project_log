@rank_zero_only
def on_train_start(self, trainer, pl_module):
    try:
        npars = pl_module.model.model.num_parameters()
    except AttributeError:
        npars = pl_module.model.num_parameters()
    n_trainable_pars = count_trainable_parameters(pl_module)
    trainer.logger.log_metrics({'n_params': npars, 'mp': npars / 1000000.0,
        'grad_mp': n_trainable_pars / 1000000.0})
