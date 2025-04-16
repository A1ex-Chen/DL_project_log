def scalings_for_boundary_conditions(timestep, sigma_data=0.5,
    timestep_scaling=1.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data ** 2 / (scaled_timestep ** 2 + sigma_data ** 2)
    c_out = scaled_timestep / (scaled_timestep ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out
