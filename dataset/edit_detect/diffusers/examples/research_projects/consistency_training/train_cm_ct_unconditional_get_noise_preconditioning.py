def get_noise_preconditioning(sigmas, noise_precond_type: str='cm'):
    """
    Calculates the noise preconditioning function c_noise, which is used to transform the raw Karras sigmas into the
    timestep input for the U-Net.
    """
    if noise_precond_type == 'none':
        return sigmas
    elif noise_precond_type == 'edm':
        return 0.25 * torch.log(sigmas)
    elif noise_precond_type == 'cm':
        return 1000 * 0.25 * torch.log(sigmas + 1e-44)
    else:
        raise ValueError(
            f'Noise preconditioning type {noise_precond_type} is not current supported. Currently supported noise preconditioning types are `none` (which uses the sigmas as is), `edm`, and `cm`.'
            )
