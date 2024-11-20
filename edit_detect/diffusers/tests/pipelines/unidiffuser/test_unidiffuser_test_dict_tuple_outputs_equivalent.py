def test_dict_tuple_outputs_equivalent(self):
    expected_slice = None
    if torch_device == 'cpu':
        expected_slice = np.array([0.7489, 0.3722, 0.4475, 0.563, 0.5923, 
            0.4992, 0.3936, 0.5844, 0.4975])
    super().test_dict_tuple_outputs_equivalent(expected_slice=expected_slice)
