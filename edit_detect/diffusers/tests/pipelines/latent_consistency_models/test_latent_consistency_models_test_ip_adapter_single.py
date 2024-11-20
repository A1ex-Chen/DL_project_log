def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.1403, 0.5072, 0.5316, 0.1202, 
            0.3865, 0.4211, 0.5363, 0.3557, 0.3645])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
