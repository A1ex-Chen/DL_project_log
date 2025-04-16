def minimize(self, loss, var_list, grad_loss=None, name=None,
    decay_var_list=None, tape=None):
    """Minimize `loss` by updating `var_list`.

        This method simply computes gradient using `tf.GradientTape` and calls
        `apply_gradients()`. If you want to process the gradient before
        applying then call `tf.GradientTape` and `apply_gradients()` explicitly
        instead of using this function.

        Args:
            loss: `Tensor` or callable. If a callable, `loss` should take no
                arguments and return the value to minimize. If a `Tensor`, the
                `tape` argument must be passed.
            var_list: list or tuple of `Variable` objects to update to
                minimize `loss`, or a callable returning the list or tuple of
                `Variable` objects. Use callable when the variable list would
                otherwise be incomplete before `minimize` since the variables
                are created at the first time `loss` is called.
            grad_loss: Optional. A `Tensor` holding the gradient computed for
                `loss`.
            decay_var_list: Optional list of variables to be decayed. Defaults
                to all variables in var_list.
            name: Optional name for the returned operation.
            tape: (Optional) `tf.GradientTape`. If `loss` is provided as a
                `Tensor`, the tape that computed the `loss` must be provided.
        Returns:
            An Operation that updates the variables in `var_list`.
        Raises:
            ValueError: If some of the variables are not `Variable` objects.
        """
    self._decay_var_list = set([v.ref() for v in decay_var_list]
        ) if decay_var_list else self._decay_var_list
    return super().minimize(loss, var_list=var_list, grad_loss=grad_loss,
        name=name, tape=tape)
