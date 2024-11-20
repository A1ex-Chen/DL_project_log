def _get_args_for_profiling(self, args, kwargs, for_inplace=False):
    cloned_args = tuple(map(lambda arg: self._clone_tensors(arg,
        for_inplace), args))
    cloned_kwargs = {key: self._clone_tensors(value, for_inplace) for key,
        value in kwargs.items()}
    return cloned_args, cloned_kwargs
