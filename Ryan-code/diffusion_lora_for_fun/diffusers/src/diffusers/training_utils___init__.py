def __init__(self, parameters: Iterable[torch.nn.Parameter], decay: float=
    0.9999, min_decay: float=0.0, update_after_step: int=0, use_ema_warmup:
    bool=False, inv_gamma: Union[float, int]=1.0, power: Union[float, int]=
    2 / 3, model_cls: Optional[Any]=None, model_config: Dict[str, Any]=None,
    **kwargs):
    """
        Args:
            parameters (Iterable[torch.nn.Parameter]): The parameters to track.
            decay (float): The decay factor for the exponential moving average.
            min_decay (float): The minimum decay factor for the exponential moving average.
            update_after_step (int): The number of steps to wait before starting to update the EMA weights.
            use_ema_warmup (bool): Whether to use EMA warmup.
            inv_gamma (float):
                Inverse multiplicative factor of EMA warmup. Default: 1. Only used if `use_ema_warmup` is True.
            power (float): Exponential factor of EMA warmup. Default: 2/3. Only used if `use_ema_warmup` is True.
            device (Optional[Union[str, torch.device]]): The device to store the EMA weights on. If None, the EMA
                        weights will be stored on CPU.

        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        """
    if isinstance(parameters, torch.nn.Module):
        deprecation_message = (
            'Passing a `torch.nn.Module` to `ExponentialMovingAverage` is deprecated. Please pass the parameters of the module instead.'
            )
        deprecate('passing a `torch.nn.Module` to `ExponentialMovingAverage`',
            '1.0.0', deprecation_message, standard_warn=False)
        parameters = parameters.parameters()
        use_ema_warmup = True
    if kwargs.get('max_value', None) is not None:
        deprecation_message = (
            'The `max_value` argument is deprecated. Please use `decay` instead.'
            )
        deprecate('max_value', '1.0.0', deprecation_message, standard_warn=
            False)
        decay = kwargs['max_value']
    if kwargs.get('min_value', None) is not None:
        deprecation_message = (
            'The `min_value` argument is deprecated. Please use `min_decay` instead.'
            )
        deprecate('min_value', '1.0.0', deprecation_message, standard_warn=
            False)
        min_decay = kwargs['min_value']
    parameters = list(parameters)
    self.shadow_params = [p.clone().detach() for p in parameters]
    if kwargs.get('device', None) is not None:
        deprecation_message = (
            'The `device` argument is deprecated. Please use `to` instead.')
        deprecate('device', '1.0.0', deprecation_message, standard_warn=False)
        self.to(device=kwargs['device'])
    self.temp_stored_params = None
    self.decay = decay
    self.min_decay = min_decay
    self.update_after_step = update_after_step
    self.use_ema_warmup = use_ema_warmup
    self.inv_gamma = inv_gamma
    self.power = power
    self.optimization_step = 0
    self.cur_decay_value = None
    self.model_cls = model_cls
    self.model_config = model_config
