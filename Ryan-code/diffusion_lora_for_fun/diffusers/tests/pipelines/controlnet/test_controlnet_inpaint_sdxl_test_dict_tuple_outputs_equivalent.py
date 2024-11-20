def test_dict_tuple_outputs_equivalent(self):
    expected_slice = None
    if torch_device == 'cpu':
        expected_slice = np.array([0.549, 0.5053, 0.4676, 0.5816, 0.5364, 
            0.483, 0.5937, 0.5719, 0.4318])
    super().test_dict_tuple_outputs_equivalent(expected_slice=expected_slice)
