def singlestep_dpm_solver_update(self, model_output_list: List[torch.Tensor
    ], *args, sample: torch.Tensor=None, order: int=None, **kwargs
    ) ->torch.Tensor:
    """
        One step for the singlestep DPMSolver.

        Args:
            model_output_list (`List[torch.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`):
                The current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`):
                The previous discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by diffusion process.
            order (`int`):
                The solver order at this step.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
    timestep_list = args[0] if len(args) > 0 else kwargs.pop('timestep_list',
        None)
    prev_timestep = args[1] if len(args) > 1 else kwargs.pop('prev_timestep',
        None)
    if sample is None:
        if len(args) > 2:
            sample = args[2]
        else:
            raise ValueError(' missing`sample` as a required keyward argument')
    if order is None:
        if len(args) > 3:
            order = args[3]
        else:
            raise ValueError(' missing `order` as a required keyward argument')
    if timestep_list is not None:
        deprecate('timestep_list', '1.0.0',
            'Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
    if prev_timestep is not None:
        deprecate('prev_timestep', '1.0.0',
            'Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`'
            )
    if order == 1:
        return self.dpm_solver_first_order_update(model_output_list[-1],
            sample=sample)
    elif order == 2:
        return self.singlestep_dpm_solver_second_order_update(model_output_list
            , sample=sample)
    elif order == 3:
        return self.singlestep_dpm_solver_third_order_update(model_output_list,
            sample=sample)
    else:
        raise ValueError(f'Order must be 1, 2, 3, got {order}')
