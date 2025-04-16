def test_dict_tuple_outputs_equivalent(self):
    expected_slice = None
    if torch_device == 'cpu':
        expected_slice = np.array([0.4051, 0.4495, 0.448, 0.5845, 0.4172, 
            0.6066, 0.4205, 0.3786, 0.5323])
    return super().test_dict_tuple_outputs_equivalent(expected_slice=
        expected_slice)
