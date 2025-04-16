def test_dict_tuple_outputs_equivalent(self):
    expected_slice = None
    if torch_device == 'cpu':
        expected_slice = np.array([0.4803, 0.3865, 0.1422, 0.6119, 0.2283, 
            0.6365, 0.5453, 0.5205, 0.3581])
    super().test_dict_tuple_outputs_equivalent(expected_slice=expected_slice)
