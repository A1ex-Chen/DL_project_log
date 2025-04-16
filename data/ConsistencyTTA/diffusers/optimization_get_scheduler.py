def get_scheduler(name: Union[str, SchedulerType], optimizer: Optimizer,
    num_warmup_steps: Optional[int]=None, num_training_steps: Optional[int]
    =None, num_cycles: int=1, power: float=1.0, last_epoch: int=-1):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_cycles (`int`, *optional*):
            The number of hard restarts used in `COSINE_WITH_RESTARTS` scheduler.
        power (`float`, *optional*, defaults to 1.0):
            Power factor. See `POLYNOMIAL` scheduler
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, last_epoch=last_epoch)
    if num_warmup_steps is None:
        raise ValueError(
            f'{name} requires `num_warmup_steps`, please provide that argument.'
            )
    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps,
            last_epoch=last_epoch)
    if num_training_steps is None:
        raise ValueError(
            f'{name} requires `num_training_steps`, please provide that argument.'
            )
    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps, num_cycles=num_cycles,
            last_epoch=last_epoch)
    if name == SchedulerType.POLYNOMIAL:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps, power=power, last_epoch=
            last_epoch)
    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps, last_epoch=last_epoch)
