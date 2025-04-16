@staticmethod
def _get_options(param_group, param_shape):
    factored = len(param_shape) >= 2
    use_first_moment = param_group['beta1'] is not None
    return factored, use_first_moment
