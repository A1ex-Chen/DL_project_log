def _calculate_lr(self) ->int:
    """Calculates the learning rate given the current step."""
    return get_scalar_from_tensor(self._get_base_optimizer()._decayed_lr(
        var_dtype=tf.float32))
