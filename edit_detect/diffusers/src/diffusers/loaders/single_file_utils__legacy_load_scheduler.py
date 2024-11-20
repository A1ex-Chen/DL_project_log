def _legacy_load_scheduler(cls, checkpoint, component_name, original_config
    =None, **kwargs):
    scheduler_type = kwargs.get('scheduler_type', None)
    prediction_type = kwargs.get('prediction_type', None)
    if scheduler_type is not None:
        deprecation_message = (
            'Please pass an instance of a Scheduler object directly to the `scheduler` argument in `from_single_file`.'
            )
        deprecate('scheduler_type', '1.0.0', deprecation_message)
    if prediction_type is not None:
        deprecation_message = (
            'Please configure an instance of a Scheduler with the appropriate `prediction_type` and pass the object directly to the `scheduler` argument in `from_single_file`.'
            )
        deprecate('prediction_type', '1.0.0', deprecation_message)
    scheduler_config = SCHEDULER_DEFAULT_CONFIG
    model_type = infer_diffusers_model_type(checkpoint=checkpoint)
    global_step = checkpoint['global_step'
        ] if 'global_step' in checkpoint else None
    if original_config:
        num_train_timesteps = getattr(original_config['model']['params'],
            'timesteps', 1000)
    else:
        num_train_timesteps = 1000
    scheduler_config['num_train_timesteps'] = num_train_timesteps
    if model_type == 'v2':
        if prediction_type is None:
            prediction_type = ('epsilon' if global_step == 875000 else
                'v_prediction')
    else:
        prediction_type = prediction_type or 'epsilon'
    scheduler_config['prediction_type'] = prediction_type
    if model_type in ['xl_base', 'xl_refiner']:
        scheduler_type = 'euler'
    elif model_type == 'playground':
        scheduler_type = 'edm_dpm_solver_multistep'
    else:
        if original_config:
            beta_start = original_config['model']['params'].get('linear_start')
            beta_end = original_config['model']['params'].get('linear_end')
        else:
            beta_start = 0.02
            beta_end = 0.085
        scheduler_config['beta_start'] = beta_start
        scheduler_config['beta_end'] = beta_end
        scheduler_config['beta_schedule'] = 'scaled_linear'
        scheduler_config['clip_sample'] = False
        scheduler_config['set_alpha_to_one'] = False
    if component_name == 'low_res_scheduler':
        return cls.from_config({'beta_end': 0.02, 'beta_schedule':
            'scaled_linear', 'beta_start': 0.0001, 'clip_sample': True,
            'num_train_timesteps': 1000, 'prediction_type': 'epsilon',
            'trained_betas': None, 'variance_type': 'fixed_small'})
    if scheduler_type is None:
        return cls.from_config(scheduler_config)
    elif scheduler_type == 'pndm':
        scheduler_config['skip_prk_steps'] = True
        scheduler = PNDMScheduler.from_config(scheduler_config)
    elif scheduler_type == 'lms':
        scheduler = LMSDiscreteScheduler.from_config(scheduler_config)
    elif scheduler_type == 'heun':
        scheduler = HeunDiscreteScheduler.from_config(scheduler_config)
    elif scheduler_type == 'euler':
        scheduler = EulerDiscreteScheduler.from_config(scheduler_config)
    elif scheduler_type == 'euler-ancestral':
        scheduler = EulerAncestralDiscreteScheduler.from_config(
            scheduler_config)
    elif scheduler_type == 'dpm':
        scheduler = DPMSolverMultistepScheduler.from_config(scheduler_config)
    elif scheduler_type == 'ddim':
        scheduler = DDIMScheduler.from_config(scheduler_config)
    elif scheduler_type == 'edm_dpm_solver_multistep':
        scheduler_config = {'algorithm_type': 'dpmsolver++',
            'dynamic_thresholding_ratio': 0.995, 'euler_at_final': False,
            'final_sigmas_type': 'zero', 'lower_order_final': True,
            'num_train_timesteps': 1000, 'prediction_type': 'epsilon',
            'rho': 7.0, 'sample_max_value': 1.0, 'sigma_data': 0.5,
            'sigma_max': 80.0, 'sigma_min': 0.002, 'solver_order': 2,
            'solver_type': 'midpoint', 'thresholding': False}
        scheduler = EDMDPMSolverMultistepScheduler(**scheduler_config)
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")
    return scheduler
