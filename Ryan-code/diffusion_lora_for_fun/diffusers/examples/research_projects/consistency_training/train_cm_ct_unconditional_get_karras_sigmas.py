def get_karras_sigmas(num_discretization_steps: int, sigma_min: float=0.002,
    sigma_max: float=80.0, rho: float=7.0, dtype=torch.float32):
    """
    Calculates the Karras sigmas timestep discretization of [sigma_min, sigma_max].
    """
    ramp = np.linspace(0, 1, num_discretization_steps)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = sigmas[::-1].copy()
    sigmas = torch.from_numpy(sigmas).to(dtype=dtype)
    return sigmas
