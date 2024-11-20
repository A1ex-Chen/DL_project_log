@torch.no_grad()
def step(self, parameters: Iterable[torch.nn.Parameter]):
    if isinstance(parameters, torch.nn.Module):
        deprecation_message = (
            'Passing a `torch.nn.Module` to `ExponentialMovingAverage.step` is deprecated. Please pass the parameters of the module instead.'
            )
        deprecate(
            'passing a `torch.nn.Module` to `ExponentialMovingAverage.step`',
            '1.0.0', deprecation_message, standard_warn=False)
        parameters = parameters.parameters()
    parameters = list(parameters)
    self.optimization_step += 1
    decay = self.get_decay(self.optimization_step)
    self.cur_decay_value = decay
    one_minus_decay = 1 - decay
    context_manager = contextlib.nullcontext
    if is_transformers_available(
        ) and transformers.deepspeed.is_deepspeed_zero3_enabled():
        import deepspeed
    for s_param, param in zip(self.shadow_params, parameters):
        if is_transformers_available(
            ) and transformers.deepspeed.is_deepspeed_zero3_enabled():
            context_manager = deepspeed.zero.GatheredParameters(param,
                modifier_rank=None)
        with context_manager():
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
            else:
                s_param.copy_(param)
