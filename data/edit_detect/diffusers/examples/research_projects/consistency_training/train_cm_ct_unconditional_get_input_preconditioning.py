def get_input_preconditioning(sigmas, sigma_data=0.5, input_precond_type:
    str='cm'):
    """
    Calculates the input preconditioning factor c_in, which is used to scale the U-Net image input.
    """
    if input_precond_type == 'none':
        return 1
    elif input_precond_type == 'cm':
        return 1.0 / (sigmas ** 2 + sigma_data ** 2)
    else:
        raise ValueError(
            f'Input preconditioning type {input_precond_type} is not current supported. Currently supported input preconditioning types are `none` (which uses a scaling factor of 1.0) and `cm`.'
            )
