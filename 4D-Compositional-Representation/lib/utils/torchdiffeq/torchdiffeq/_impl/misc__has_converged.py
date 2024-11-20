def _has_converged(y0, y1, rtol, atol):
    """Checks that each element is within the error tolerance."""
    error_tol = tuple(atol + rtol * torch.max(torch.abs(y0_), torch.abs(y1_
        )) for y0_, y1_ in zip(y0, y1))
    error = tuple(torch.abs(y0_ - y1_) for y0_, y1_ in zip(y0, y1))
    return all((error_ < error_tol_).all() for error_, error_tol_ in zip(
        error, error_tol))
