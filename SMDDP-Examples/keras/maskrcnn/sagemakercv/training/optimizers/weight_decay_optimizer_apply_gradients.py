def apply_gradients(self, grads_and_vars, name=None, decay_var_list=None,
    **kwargs):
    """Apply gradients to variables.

        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.

        Args:
            grads_and_vars: List of (gradient, variable) pairs.
            name: Optional name for the returned operation.  Default to the
                name passed to the `Optimizer` constructor.
            decay_var_list: Optional list of variables to be decayed. Defaults
                to all variables in var_list.
            **kwargs: Additional arguments to pass to the base optimizer's
                apply_gradient method, e.g., TF2.2 added an argument
                `experimental_aggregate_gradients`.
        Returns:
            An `Operation` that applies the specified gradients.
        Raises:
            TypeError: If `grads_and_vars` is malformed.
            ValueError: If none of the variables have gradients.
        """
    self._decay_var_list = set([v.ref() for v in decay_var_list]
        ) if decay_var_list else self._decay_var_list
    return super().apply_gradients(grads_and_vars, name=name, **kwargs)
