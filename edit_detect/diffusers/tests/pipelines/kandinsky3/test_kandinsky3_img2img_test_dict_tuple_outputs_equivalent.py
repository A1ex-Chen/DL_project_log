def test_dict_tuple_outputs_equivalent(self):
    expected_slice = None
    if torch_device == 'cpu':
        expected_slice = np.array([0.5762, 0.6112, 0.415, 0.6018, 0.6167, 
            0.4626, 0.5426, 0.5641, 0.6536])
    super().test_dict_tuple_outputs_equivalent(expected_slice=expected_slice)
