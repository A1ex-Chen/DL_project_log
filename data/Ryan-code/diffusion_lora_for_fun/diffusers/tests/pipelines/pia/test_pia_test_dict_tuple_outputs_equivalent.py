def test_dict_tuple_outputs_equivalent(self):
    expected_slice = None
    if torch_device == 'cpu':
        expected_slice = np.array([0.374, 0.4284, 0.4038, 0.5417, 0.4405, 
            0.5521, 0.4273, 0.4124, 0.4997])
    return super().test_dict_tuple_outputs_equivalent(expected_slice=
        expected_slice)
