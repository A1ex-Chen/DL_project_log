def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__
            .__name__, unused_kwargs))
