def test_dict_tuple_outputs_equivalent(self):
    expected_slice = None
    if torch_device == 'cpu':
        expected_slice = np.array([0.6391, 0.629, 0.486, 0.5134, 0.555, 
            0.4577, 0.5033, 0.5023, 0.4538])
    super().test_dict_tuple_outputs_equivalent(expected_slice=
        expected_slice, expected_max_difference=0.003)
