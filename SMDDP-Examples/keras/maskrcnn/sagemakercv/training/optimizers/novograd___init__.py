@typechecked
def __init__(self, learning_rate: Union[FloatTensorLike, Callable]=0.001,
    beta_1: FloatTensorLike=0.9, beta_2: FloatTensorLike=0.999, epsilon:
    FloatTensorLike=1e-07, weight_decay: FloatTensorLike=0.0,
    exclude_from_weight_decay: Optional[List[str]]=None, grad_averaging:
    bool=False, amsgrad: bool=False, name: str='NovoGrad', **kwargs):
    """Construct a new NovoGrad optimizer.

        Args:
            learning_rate: A `Tensor` or a floating point value. or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                The learning rate.
            beta_1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay: A floating point value. Weight decay for each param.
            grad_averaging: determines whether to use Adam style exponential
                moving averaging for the first order moments.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
    super().__init__(name, **kwargs)
    if weight_decay < 0.0:
        raise ValueError('Weight decay rate cannot be negative')
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('decay', self._initial_decay)
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self._set_hyper('weight_decay', weight_decay)
    self._set_hyper('grad_averaging', grad_averaging)
    self.amsgrad = amsgrad
    self.epsilon = epsilon or tf.keras.backend.epsilon()
    self.exclude_from_weight_decay = exclude_from_weight_decay
