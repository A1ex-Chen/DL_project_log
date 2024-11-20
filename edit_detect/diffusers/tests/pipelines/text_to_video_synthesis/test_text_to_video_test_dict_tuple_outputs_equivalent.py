def test_dict_tuple_outputs_equivalent(self):
    expected_slice = None
    if torch_device == 'cpu':
        expected_slice = np.array([0.4903, 0.5649, 0.5504, 0.5179, 0.4821, 
            0.5466, 0.4131, 0.5052, 0.5077])
    return super().test_dict_tuple_outputs_equivalent(expected_slice=
        expected_slice)
