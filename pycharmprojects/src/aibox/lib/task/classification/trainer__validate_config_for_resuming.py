@staticmethod
def _validate_config_for_resuming(config: Config, checkpoint: Checkpoint):
    for group in checkpoint.optimizer.param_groups:
        assert config.learning_rate == group['initial_lr'
            ], 'config.learning_rate is inconsistent with resuming one: {} vs {}'.format(
            config.learning_rate, group['initial_lr'])
        assert config.momentum == group['momentum'
            ], 'config.momentum is inconsistent with resuming one: {} vs {}'.format(
            config.momentum, group['momentum'])
        assert config.weight_decay == group['weight_decay'
            ], 'config.weight_decay is inconsistent with resuming one: {} vs {}'.format(
            config.weight_decay, group['weight_decay'])
    assert config.pretrained == checkpoint.model.algorithm.pretrained, 'config.pretrained is inconsistent with resuming one: {} vs {}'.format(
        config.pretrained, checkpoint.model.algorithm.pretrained)
