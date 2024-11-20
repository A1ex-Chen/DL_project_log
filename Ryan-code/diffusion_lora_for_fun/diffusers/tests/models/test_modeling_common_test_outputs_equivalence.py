def test_outputs_equivalence(self):

    def set_nan_tensor_to_zero(t):
        device = t.device
        if device.type == 'mps':
            t = t.to('cpu')
        t[t != t] = 0
        return t.to(device)

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
    if self.forward_requires_fresh_args:
        model = self.model_class(**self.init_dict)
    else:
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
    model.to(torch_device)
    model.eval()
    with torch.no_grad():
        if self.forward_requires_fresh_args:
            outputs_dict = model(**self.inputs_dict(0))
            outputs_tuple = model(**self.inputs_dict(0), return_dict=False)
        else:
            outputs_dict = model(**inputs_dict)
            outputs_tuple = model(**inputs_dict, return_dict=False)
    recursive_check(outputs_tuple, outputs_dict)
