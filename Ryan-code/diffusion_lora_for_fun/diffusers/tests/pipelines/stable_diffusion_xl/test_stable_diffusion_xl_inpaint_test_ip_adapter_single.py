def test_ip_adapter_single(self):
    expected_pipe_slice = None
    if torch_device == 'cpu':
        expected_pipe_slice = np.array([0.7971, 0.5371, 0.5973, 0.5642, 
            0.6689, 0.6894, 0.577, 0.6063, 0.5261])
    return super().test_ip_adapter_single(expected_pipe_slice=
        expected_pipe_slice)
