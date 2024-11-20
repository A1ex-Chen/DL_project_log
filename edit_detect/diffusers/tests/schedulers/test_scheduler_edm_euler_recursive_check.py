def recursive_check(tuple_object, dict_object):
    if isinstance(tuple_object, (List, Tuple)):
        for tuple_iterable_value, dict_iterable_value in zip(tuple_object,
            dict_object.values()):
            recursive_check(tuple_iterable_value, dict_iterable_value)
    elif isinstance(tuple_object, Dict):
        for tuple_iterable_value, dict_iterable_value in zip(tuple_object.
            values(), dict_object.values()):
            recursive_check(tuple_iterable_value, dict_iterable_value)
    elif tuple_object is None:
        return
    else:
        self.assertTrue(torch.allclose(set_nan_tensor_to_zero(tuple_object),
            set_nan_tensor_to_zero(dict_object), atol=1e-05), msg=
            f'Tuple and dict output are not equal. Difference: {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`: {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}.'
            )
