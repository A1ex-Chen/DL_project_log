def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """DP-SGD version of base class method."""
    assert self._was_compute_gradients_called, 'compute_gradients() on the differentially private optimizer was not called. Which means that the training is not differentially private. It happens for example in Keras training in TensorFlow 2.0+.'
    return super(DPOptimizerClass, self).apply_gradients(grads_and_vars=
        grads_and_vars, global_step=global_step, name=name)
