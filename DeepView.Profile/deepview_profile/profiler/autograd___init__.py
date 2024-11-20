def __init__(self, grad_fn_ordering, input_map, initial_inputs):
    self._grad_fn_ordering = grad_fn_ordering
    self._input_holder = {fn: ([None] * size) for fn, size in input_map.items()
        }
    self._input_holder[self._grad_fn_ordering[0]] = initial_inputs
