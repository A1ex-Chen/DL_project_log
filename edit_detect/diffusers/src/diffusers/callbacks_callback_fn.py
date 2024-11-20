def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) ->Dict[
    str, Any]:
    cutoff_step_ratio = self.config.cutoff_step_ratio
    cutoff_step_index = self.config.cutoff_step_index
    cutoff_step = cutoff_step_index if cutoff_step_index is not None else int(
        pipeline.num_timesteps * cutoff_step_ratio)
    if step_index == cutoff_step:
        pipeline.set_ip_adapter_scale(0.0)
    return callback_kwargs
