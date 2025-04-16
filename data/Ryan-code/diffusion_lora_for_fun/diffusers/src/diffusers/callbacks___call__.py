def __call__(self, pipeline, step_index, timestep, callback_kwargs) ->Dict[
    str, Any]:
    """
        Calls all the callbacks in order with the given arguments and returns the final callback_kwargs.
        """
    for callback in self.callbacks:
        callback_kwargs = callback(pipeline, step_index, timestep,
            callback_kwargs)
    return callback_kwargs
