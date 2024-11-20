def singlestep_dpm_solver_update(self, model_output_list: List[torch.
    FloatTensor], timestep_list: List[int], prev_timestep: int, sample:
    torch.FloatTensor, order: int) ->torch.FloatTensor:
    """
        One step for the singlestep DPM-Solver.

        Args:
            model_output_list (`List[torch.FloatTensor]`):
                direct outputs from learned diffusion model at current and latter timesteps.
            timestep (`int`): current and latter discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            order (`int`):
                the solver order at this step.

        Returns:
            `torch.FloatTensor`: the sample tensor at the previous timestep.
        """
    if order == 1:
        return self.dpm_solver_first_order_update(model_output_list[-1],
            timestep_list[-1], prev_timestep, sample)
    elif order == 2:
        return self.singlestep_dpm_solver_second_order_update(model_output_list
            , timestep_list, prev_timestep, sample)
    elif order == 3:
        return self.singlestep_dpm_solver_third_order_update(model_output_list,
            timestep_list, prev_timestep, sample)
    else:
        raise ValueError(f'Order must be 1, 2, 3, got {order}')
