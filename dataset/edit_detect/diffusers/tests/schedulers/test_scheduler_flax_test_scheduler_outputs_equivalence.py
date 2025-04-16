def test_scheduler_outputs_equivalence(self):

    def set_nan_tensor_to_zero(t):
        return t.at[t != t].set(0)

    def recursive_check(tuple_object, dict_object):
        if isinstance(tuple_object, (List, Tuple)):
            for tuple_iterable_value, dict_iterable_value in zip(tuple_object,
                dict_object.values()):
                recursive_check(tuple_iterable_value, dict_iterable_value)
        elif isinstance(tuple_object, Dict):
            for tuple_iterable_value, dict_iterable_value in zip(tuple_object
                .values(), dict_object.values()):
                recursive_check(tuple_iterable_value, dict_iterable_value)
        elif tuple_object is None:
            return
        else:
            self.assertTrue(jnp.allclose(set_nan_tensor_to_zero(
                tuple_object), set_nan_tensor_to_zero(dict_object), atol=
                1e-05), msg=
                f'Tuple and dict output are not equal. Difference: {jnp.max(jnp.abs(tuple_object - dict_object))}. Tuple has `nan`: {jnp.isnan(tuple_object).any()} and `inf`: {jnp.isinf(tuple_object)}. Dict has `nan`: {jnp.isnan(dict_object).any()} and `inf`: {jnp.isinf(dict_object)}.'
                )
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', None)
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.create_state()
        sample, _ = self.dummy_sample
        residual = 0.1 * sample
        if num_inference_steps is not None and hasattr(scheduler,
            'set_timesteps'):
            state = scheduler.set_timesteps(state, num_inference_steps,
                shape=sample.shape)
        elif num_inference_steps is not None and not hasattr(scheduler,
            'set_timesteps'):
            kwargs['num_inference_steps'] = num_inference_steps
        outputs_dict = scheduler.step(state, residual, 0, sample, **kwargs)
        if num_inference_steps is not None and hasattr(scheduler,
            'set_timesteps'):
            state = scheduler.set_timesteps(state, num_inference_steps,
                shape=sample.shape)
        elif num_inference_steps is not None and not hasattr(scheduler,
            'set_timesteps'):
            kwargs['num_inference_steps'] = num_inference_steps
        outputs_tuple = scheduler.step(state, residual, 0, sample,
            return_dict=False, **kwargs)
        recursive_check(outputs_tuple[0], outputs_dict.prev_sample)
