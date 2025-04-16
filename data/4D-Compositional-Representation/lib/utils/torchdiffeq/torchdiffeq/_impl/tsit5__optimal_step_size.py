def _optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=
    10.0, dfactor=0.2, order=5):
    """Calculate the optimal size for the next Runge-Kutta step."""
    if mean_error_ratio == 0:
        return last_step * ifactor
    if mean_error_ratio < 1:
        dfactor = _convert_to_tensor(1, dtype=torch.float64, device=
            mean_error_ratio.device)
    error_ratio = torch.sqrt(mean_error_ratio).type_as(last_step)
    exponent = torch.tensor(1 / order).type_as(last_step)
    factor = torch.max(1 / ifactor, torch.min(error_ratio ** exponent /
        safety, 1 / dfactor))
    return last_step / factor
