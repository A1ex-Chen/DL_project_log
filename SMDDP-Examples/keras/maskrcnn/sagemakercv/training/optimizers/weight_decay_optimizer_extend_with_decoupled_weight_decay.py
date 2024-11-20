@typechecked
def extend_with_decoupled_weight_decay(base_optimizer: Type[tf.keras.
    optimizers.Optimizer]) ->Type[tf.keras.optimizers.Optimizer]:
    """Factory function returning an optimizer class with decoupled weight
    decay.

    Returns an optimizer class. An instance of the returned class computes the
    update step of `base_optimizer` and additionally decays the weights.
    E.g., the class returned by
    `extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)` is
    equivalent to `tfa.optimizers.AdamW`.

    The API of the new optimizer class slightly differs from the API of the
    base optimizer:
    - The first argument to the constructor is the weight decay rate.
    - `minimize` and `apply_gradients` accept the optional keyword argument
      `decay_var_list`, which specifies the variables that should be decayed.
      If `None`, all variables that are optimized are decayed.

    Usage example:
    ```python
    # MyAdamW is a new class
    MyAdamW = extend_with_decoupled_weight_decay(tf.keras.optimizers.Adam)
    # Create a MyAdamW object
    optimizer = MyAdamW(weight_decay=0.001, learning_rate=0.001)
    # update var1, var2 but only decay var1
    optimizer.minimize(loss, var_list=[var1, var2], decay_variables=[var1])

    Note: this extension decays weights BEFORE applying the update based
    on the gradient, i.e. this extension only has the desired behaviour for
    optimizers which do not depend on the value of 'var' in the update step!

    Note: when applying a decay to the learning rate, be sure to manually apply
    the decay to the `weight_decay` as well. For example:

    ```python
    step = tf.Variable(0, trainable=False)
    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [10000, 15000], [1e-0, 1e-1, 1e-2])
    # lr and wd can be a function or a tensor
    lr = 1e-1 * schedule(step)
    wd = lambda: 1e-4 * schedule(step)

    # ...

    optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    ```

    Note: you might want to register your own custom optimizer using
    `tf.keras.utils.get_custom_objects()`.

    Args:
        base_optimizer: An optimizer class that inherits from
            tf.optimizers.Optimizer.

    Returns:
        A new optimizer class that inherits from DecoupledWeightDecayExtension
        and base_optimizer.
    """


    class OptimizerWithDecoupledWeightDecay(DecoupledWeightDecayExtension,
        base_optimizer):
        """Base_optimizer with decoupled weight decay.

        This class computes the update step of `base_optimizer` and
        additionally decays the variable with the weight decay being
        decoupled from the optimization steps w.r.t. to the loss
        function, as described by Loshchilov & Hutter
        (https://arxiv.org/pdf/1711.05101.pdf). For SGD variants, this
        simplifies hyperparameter search since it decouples the settings
        of weight decay and learning rate. For adaptive gradient
        algorithms, it regularizes variables with large gradients more
        than L2 regularization would, which was shown to yield better
        training loss and generalization error in the paper above.
        """

        @typechecked
        def __init__(self, weight_decay: Union[FloatTensorLike, Callable],
            decay_var_list: Optional[List]=None, *args, **kwargs):
            super().__init__(weight_decay, decay_var_list, *args, **kwargs)
    return OptimizerWithDecoupledWeightDecay
