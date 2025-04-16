def _log_scalars(scalars, step=0):
    """Log scalars to the NeptuneAI experiment logger."""
    if run:
        for k, v in scalars.items():
            run[k].append(value=v, step=step)
