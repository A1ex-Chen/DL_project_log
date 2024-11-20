def test_dict_tuple_outputs_equivalent(self):
    expected_slice = None
    if torch_device == 'cpu':
        expected_slice = np.array([0.5052, 0.5546, 0.4567, 0.477, 0.5195, 
            0.4085, 0.5026, 0.4909, 0.4495])
    super().test_dict_tuple_outputs_equivalent(expected_slice=expected_slice)
