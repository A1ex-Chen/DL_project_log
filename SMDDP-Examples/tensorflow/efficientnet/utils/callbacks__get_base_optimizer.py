def _get_base_optimizer(self) ->tf.keras.optimizers.Optimizer:
    """Get the base optimizer used by the current model."""
    optimizer = self.model.optimizer
    while hasattr(optimizer, '_optimizer'):
        optimizer = optimizer._optimizer
    return optimizer
