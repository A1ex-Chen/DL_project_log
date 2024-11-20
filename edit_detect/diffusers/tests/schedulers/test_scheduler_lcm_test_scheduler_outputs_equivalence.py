def test_scheduler_outputs_equivalence(self):

    def set_nan_tensor_to_zero(t):
        t[t != t] = 0
        return t

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
            self.assertTrue(torch.allclose(set_nan_tensor_to_zero(
                tuple_object), set_nan_tensor_to_zero(dict_object), atol=
                1e-05), msg=
                f'Tuple and dict output are not equal. Difference: {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`: {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}.'
                )
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.pop('num_inference_steps', 50)
    timestep = self.default_valid_timestep
    for scheduler_class in self.scheduler_classes:
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        sample = self.dummy_sample
        residual = 0.1 * sample
        scheduler.set_timesteps(num_inference_steps)
        kwargs['generator'] = torch.manual_seed(0)
        outputs_dict = scheduler.step(residual, timestep, sample, **kwargs)
        scheduler.set_timesteps(num_inference_steps)
        kwargs['generator'] = torch.manual_seed(0)
        outputs_tuple = scheduler.step(residual, timestep, sample,
            return_dict=False, **kwargs)
        recursive_check(outputs_tuple, outputs_dict)
