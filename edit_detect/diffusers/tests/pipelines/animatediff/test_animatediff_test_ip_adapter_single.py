def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.5541, 0.5802, 0.5074, 0.4583, 
            0.4729, 0.5374, 0.4051, 0.4495, 0.448, 0.5292, 0.6322, 0.6265, 
            0.5455, 0.4771, 0.5795, 0.5845, 0.4172, 0.6066, 0.6535, 0.4113,
            0.6833, 0.5736, 0.3589, 0.573, 0.4205, 0.3786, 0.5323])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
