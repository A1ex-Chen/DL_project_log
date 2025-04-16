def compute_noise(scheduler, *args):
    if isinstance(scheduler, DDIMScheduler):
        return compute_noise_ddim(scheduler, *args)
    elif isinstance(scheduler, DPMSolverMultistepScheduler
        ) and scheduler.config.algorithm_type == 'sde-dpmsolver++' and scheduler.config.solver_order == 2:
        return compute_noise_sde_dpm_pp_2nd(scheduler, *args)
    else:
        raise NotImplementedError
