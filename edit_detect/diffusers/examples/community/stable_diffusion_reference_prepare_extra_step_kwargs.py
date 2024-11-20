def prepare_extra_step_kwargs(self, generator: Union[torch.Generator, List[
    torch.Generator]], eta: float) ->Dict[str, Any]:
    """
        Prepare extra keyword arguments for the scheduler step.

        Args:
            generator (Union[torch.Generator, List[torch.Generator]]): The generator used for sampling.
            eta (float): The value of eta (η) used with the DDIMScheduler. Should be between 0 and 1.

        Returns:
            Dict[str, Any]: A dictionary containing the extra keyword arguments for the scheduler step.
        """
    accepts_eta = 'eta' in set(inspect.signature(self.scheduler.step).
        parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs['eta'] = eta
    accepts_generator = 'generator' in set(inspect.signature(self.scheduler
        .step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs['generator'] = generator
    return extra_step_kwargs
