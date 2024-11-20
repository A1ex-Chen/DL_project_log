def _compute_error_ratio(error_estimate, error_tol=None, rtol=None, atol=
    None, y0=None, y1=None):
    if error_tol is None:
        assert rtol is not None and atol is not None and y0 is not None and y1 is not None
        rtol if _is_iterable(rtol) else [rtol] * len(y0)
        atol if _is_iterable(atol) else [atol] * len(y0)
        error_tol = tuple(atol_ + rtol_ * torch.max(torch.abs(y0_), torch.
            abs(y1_)) for atol_, rtol_, y0_, y1_ in zip(atol, rtol, y0, y1))
    error_ratio = tuple(error_estimate_ / error_tol_ for error_estimate_,
        error_tol_ in zip(error_estimate, error_tol))
    mean_sq_error_ratio = tuple(torch.mean(error_ratio_ * error_ratio_) for
        error_ratio_ in error_ratio)
    return mean_sq_error_ratio
